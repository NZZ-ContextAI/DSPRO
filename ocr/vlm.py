"""Thin wrapper around a Hugging Face vision-language model (Qwen-VL family) for local GPU use."""
from PIL import Image

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


class VLM:
    def __init__(self, model_id: str = DEFAULT_MODEL, load_4bit: bool = False):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        try:  # picks up HF_TOKEN from .env; on the hub the variable is usually exported instead
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        kwargs = {"device_map": "cuda", "dtype": torch.bfloat16}
        if load_4bit:  # lets ~32B models fit a 24 GB GPU
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.model_id = model_id
        self.repetition_penalty = 1.05
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()

    def generate(self, prompt: str, images: list = (), max_new_tokens: int = 4096,
                 repetition_penalty: float = None) -> str:
        """Greedy decoding; ``images`` are PIL images placed before the prompt."""
        import torch

        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=list(images) or None, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      repetition_penalty=repetition_penalty or self.repetition_penalty)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()
