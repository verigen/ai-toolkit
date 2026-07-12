import base64
import io
import os
from collections import OrderedDict
from typing import Optional

import requests

from .BaseCaptioner import BaseCaptioner, CaptionConfig


class OpenAICaptionConfig(CaptionConfig):
    def __init__(self, **kwargs):
        # model_name_or_path is reused as the API model name (e.g. "gpt-4o").
        super().__init__(**kwargs)
        self.api_base_url = kwargs.get("api_base_url", "http://localhost:8080/v1")
        # Global setting only, same as HF_TOKEN — never part of job config / on disk.
        self.api_key = os.environ.get("OPENAI_API_KEY") or None
        self.request_timeout = kwargs.get("request_timeout", 120)


class OpenAICaptioner(BaseCaptioner):
    caption_config_class = OpenAICaptionConfig

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)

    def load_model(self):
        # No local model to load; validate config and leave self.model = None
        # (so maybe_compile_models() skips cleanly).
        if not self.caption_config.api_base_url:
            raise ValueError("api_base_url is required for OpenAICaptioner")
        self.print_and_status_update(
            f"Using OpenAI-compatible API at {self.caption_config.api_base_url} "
            f"(model: {self.caption_config.model_name_or_path})"
        )
        self.model = None

    def _encode_image(self, img) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def get_caption_for_file(self, file_path: str) -> Optional[str]:
        img = self.load_pil_image(file_path, max_res=self.caption_config.max_res)
        b64 = self._encode_image(img)

        url = self.caption_config.api_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.caption_config.api_key:
            headers["Authorization"] = f"Bearer {self.caption_config.api_key}"

        payload = {
            "model": self.caption_config.model_name_or_path,
            "max_tokens": self.caption_config.max_new_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": self.caption_config.caption_prompt},
                    ],
                }
            ],
        }

        resp = requests.post(
            url, headers=headers, json=payload,
            timeout=self.caption_config.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
