from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .base import LLMCallerInterface
from pydantic_dataclasses import TokenUsage, LLMResponse

# basically stolen straight from the HF example

class HFModels:
    Qwen_2_5__3B = "Qwen/Qwen2.5-3B-Instruct"
    Qwen_2_5__0_5B = "Qwen/Qwen2.5-0.5B-Instruct"
    Qwen_2_5__1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
    Llama_3_2__3B = "meta-llama/Llama-3.2-3B-Instruct"


MODEL = HFModels.Qwen_2_5__3B


class LocalQwenLLMCaller(LLMCallerInterface):
    """Using Hugging Face Transformers with a low-VRAM loading strategy."""

    def __init__(
        self,
        model_name: str = MODEL,
        max_new_tokens: int = 512,
        temperature: float = 0.05,
        low_vram: bool = True,
        use_4bit: bool | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.low_vram = low_vram
        self.use_4bit = use_4bit
        self.cache_folder = Path("./_local_models")
        self.cache_folder.mkdir(exist_ok=True, parents=True)

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=str(self.cache_folder))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_folder),
            **self._model_loading_kwargs(),
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

    def _model_loading_kwargs(self):
        kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
        }

        if self.device == "cuda":
            if self.use_4bit is None:
                self.use_4bit = self.low_vram
            if self.use_4bit:
                try:
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    kwargs["device_map"] = "auto"
                    return kwargs
                except Exception:
                    self.use_4bit = False

            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
            return kwargs

        if self.device == "mps":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = {"": self.device}
            return kwargs

        kwargs["torch_dtype"] = torch.float32
        kwargs["device_map"] = {"": "cpu"}
        return kwargs

    def call(self, prompt: str, system_prompt: str | None = None, **kwargs) -> LLMResponse:
        # Format input using model's chat template
        system_prompt = system_prompt if system_prompt else "You are a helpful assistant."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "do_sample": kwargs.get("temperature", self.temperature) > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        outputs = self.pipe(
            formatted_prompt,
            **generation_kwargs,
        )

        generated_text = outputs[0]["generated_text"]
        response = generated_text[len(formatted_prompt):].strip()

        usage = TokenUsage(
            prompt_tokens=len(self.tokenizer.encode(formatted_prompt)),
            completion_tokens=len(self.tokenizer.encode(response, add_special_tokens=False)),
        )

        return LLMResponse(response=response, token_usage=usage)
