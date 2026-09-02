from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .base import LLMCallerInterface

# basically stolen straight from the HF example

class LocalQwenLLMCaller(LLMCallerInterface):
    """Using Hugging Face Transformers"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.05
    ):
        super().__init__()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.cache_folder = Path("./_local_models")
        self.cache_folder.mkdir(exist_ok=True, parents=True)

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_folder=self.cache_folder)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def call(self, prompt: str, **kwargs) -> str:
        # Format input using model's chat template
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self.pipe(
            formatted_prompt,
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            do_sample=True if self.temperature > 0 else False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_text = outputs[0]["generated_text"]
        response = generated_text[len(formatted_prompt):].strip()

        return response
