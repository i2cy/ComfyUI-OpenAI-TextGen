import base64
import json
from io import BytesIO
import wave

import numpy as np
import requests
from comfy import model_management
from comfy.utils import ProgressBar
from comfy_api.latest import ComfyExtension, io
from PIL import Image
from server import PromptServer
from typing_extensions import override

MAX_VIDEO_FRAMES = 32
VIDEO_FPS = 24
VIDEO_TARGET_FPS = 1
THINK_OPEN = "<" + "|think|" + ">"
THINK_CLOSE = "<" + "|/think|" + ">"


def tensor_to_png_data_url(img):
    arr = (img.cpu().numpy().squeeze() * 255.0).clip(0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def image_parts(images):
    return [{"type": "image_url", "image_url": {"url": tensor_to_png_data_url(frame)}} for frame in images]


def video_parts(video):
    if video.ndim == 5:
        video = video.reshape(-1, *video.shape[-3:])
    frames = video[::VIDEO_FPS // VIDEO_TARGET_FPS]
    if len(frames) > MAX_VIDEO_FRAMES:
        frames = frames[::(len(frames) + MAX_VIDEO_FRAMES - 1) // MAX_VIDEO_FRAMES]
    return image_parts(frames)


def audio_part(audio, sample_rate):
    data = audio.cpu().numpy()
    data = data[0] if data.ndim == 3 else data
    if data.ndim == 2 and data.shape[0] < data.shape[1]:
        data = data.T
    pcm = (data * 32767.0).clip(-32767, 32767).astype("<i2")
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}}


class OpenAITextGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        sampling_options = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.000001),
                    io.Int.Input("top_k", default=64, min=0, max=1000, tooltip="Sent as a vLLM/llama.cpp extension; omitted when 0."),
                    io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("min_p", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Sent as a vLLM extension; omitted when 0. Rejected by servers with speculative decoding enabled."),
                    io.Float.Input("repetition_penalty", default=1.0, min=0.0, max=5.0, step=0.01, tooltip="Sent as a vLLM/llama.cpp extension; omitted when 1.0."),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                    io.Float.Input("presence_penalty", optional=True, default=0.0, min=-2.0, max=2.0, step=0.01),
                ]
            ),
            io.DynamicCombo.Option(
                key="off",
                inputs=[]
            ),
        ]

        return io.Schema(
            node_id="OpenAITextGenerate",
            display_name="Generate Text (OpenAI API)",
            category="text",
            search_aliases=["LLM", "openai", "api", "vllm", "gpt", "vision", "multimodal"],
            description=(
                "Replaces the local Generate Text node with an OpenAI-compatible chat completions API: "
                "any server (vLLM, llama.cpp, Ollama, OpenAI) that serves /v1/chat/completions. "
                "Images and video frames are sent as base64 image_url parts, audio as a WAV input_audio part. "
                "Video is assumed to be 24 FPS and subsampled to 1 FPS, max 32 frames. "
                "Thinking uses vLLM's chat_template_kwargs and reads reasoning/reasoning_content from the response. "
                "The request is streamed so ComfyUI Cancel aborts generation mid-run and disconnects the server."
            ),
            inputs=[
                io.String.Input("base_url", default="http://192.168.11.104:18300/v1", tooltip="OpenAI-compatible API root ending in /v1."),
                io.String.Input("api_key", default="EMPTY", advanced=True),
                io.String.Input("model", default="qwen3.8-flash-next"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input("system_prompt", multiline=True, force_input=True, optional=True, default=""),
                io.Image.Input("image", optional=True),
                io.Image.Input("video", optional=True, tooltip="Video frames as image batch. Assumed to be 24 FPS; subsampled to 1 FPS."),
                io.Audio.Input("audio", optional=True),
                io.Int.Input("sample_rate", optional=True, default=16000, min=1000, max=192000, advanced=True, tooltip="Sample rate assumed when encoding the AUDIO tensor to WAV."),
                io.Int.Input("max_length", default=512, min=1, max=131072, tooltip="max_tokens for the request."),
                io.DynamicCombo.Input("sampling_mode", options=sampling_options, display_name="Sampling Mode"),
                io.Boolean.Input("thinking", optional=True, default=False, tooltip="Operate in thinking mode if the model supports it."),
                io.Int.Input("timeout", optional=True, default=600, min=10, max=3600, advanced=True, tooltip="Request timeout in seconds."),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
                io.String.Output(display_name="thinking"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, base_url, model, prompt, max_length, sampling_mode, api_key="EMPTY", system_prompt="", image=None, video=None, audio=None, sample_rate=16000, thinking=False, timeout=600) -> io.NodeOutput:
        content = []
        if image is not None:
            content.extend(image_parts(image))
        if video is not None:
            content.extend(video_parts(video))
        if audio is not None:
            content.append(audio_part(audio, sample_rate))
        if prompt.strip():
            content.append({"type": "text", "text": prompt})

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": model,
            "max_tokens": max_length,
            "stream": True,
            "messages": messages,
            "chat_template_kwargs": {"thinking": bool(thinking)},
        }
        if sampling_mode.get("sampling_mode") == "on":
            payload["temperature"] = sampling_mode.get("temperature", 1.0)
            payload["top_p"] = sampling_mode.get("top_p", 1.0)
            payload["seed"] = sampling_mode.get("seed", 0)
            top_k = sampling_mode.get("top_k", 0)
            if top_k > 0:
                payload["top_k"] = top_k
            min_p = sampling_mode.get("min_p", 0.0)
            if min_p > 0:
                payload["min_p"] = min_p
            repetition_penalty = sampling_mode.get("repetition_penalty", 1.0)
            if repetition_penalty != 1.0:
                payload["repetition_penalty"] = repetition_penalty
            presence_penalty = sampling_mode.get("presence_penalty", 0.0)
            if presence_penalty:
                payload["presence_penalty"] = presence_penalty
        else:
            payload["temperature"] = 0.0

        url = base_url.strip().rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        model_management.throw_exception_if_processing_interrupted()
        progress = ProgressBar(max_length, node_id=cls.hidden.unique_id)
        PromptServer.instance.send_progress_text("Waiting for API response…", cls.hidden.unique_id)

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout, stream=True)
        except requests.RequestException as exc:
            raise RuntimeError(f"OpenAI API request to {url} failed: {exc}")

        text_parts = []
        reasoning_parts = []
        text_chars = 0
        chunked = False
        first_line = ""
        try:
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text[:600]}")
            for line in resp.iter_lines(decode_unicode=True):
                first_line = first_line or line
                # Raises when ComfyUI Cancel is pressed; the finally below drops the
                # connection so the server stops generating instead of running to completion.
                model_management.throw_exception_if_processing_interrupted()
                if not line.startswith("data:"):
                    continue
                chunked = True
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    text_parts.append(delta["content"])
                    text_chars += len(delta["content"])
                delta_reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if delta_reasoning:
                    reasoning_parts.append(delta_reasoning)
                    text_chars += len(delta_reasoning)
                # ~3 chars/token estimate, capped below the total so only the final
                # update sends 100%; ProgressBar throttles the sends itself.
                progress.update_absolute(min(text_chars // 3, max_length - 1))
        finally:
            resp.close()

        progress.update_absolute(max_length)

        text = "".join(text_parts)
        reasoning = "".join(reasoning_parts)
        if not chunked:
            # Server ignored "stream"; the body is a single chat completion JSON.
            data = json.loads(first_line or "{}")
            if not data.get("choices"):
                raise RuntimeError(f"OpenAI API returned no choices: {first_line[:600]}")
            message = data["choices"][0].get("message") or {}
            text = message.get("content") or ""
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        if not reasoning and THINK_OPEN in text:
            reasoning, _, rest = text.partition(THINK_CLOSE)
            reasoning = reasoning.partition(THINK_OPEN)[2]
            text = rest

        return io.NodeOutput(text.strip(), reasoning.strip())


class OpenAITextgenExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            OpenAITextGenerate,
        ]


async def comfy_entrypoint() -> OpenAITextgenExtension:
    return OpenAITextgenExtension()
