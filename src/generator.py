"""Generate answers from sourced prompts unsing Qwen on CPU"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompting import RagPrompt


class Generator:
    """Load a language model once and reuse it for answer generation."""

    def __init__(
            self,
            model_name: str = "Qwen/Qwen3-0.6B",
            context_budget: int = 4096,
            max_new_tokens: int = 256,
    ) -> None:
        """Load the tokenizer and model with validated token limits."""
        if context_budget <= 0:
            raise ValueError("context_budget must be positive")

        if not 0 < max_new_tokens < context_budget:
            raise ValueError(
                "max_new_tokens must be positive and below context_budget"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
        )
        self.model.cpu()
        self.model.eval()

        model_limit = self.model.config.max_position_embeddings

        if context_budget > model_limit:
            raise ValueError(
                "context_budget exceeds the model context limit"
            )

        self.context_budget = context_budget
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: RagPrompt) -> str:
        """Generate an answer after checking the complete input budget."""
        messages = [
            message.model_dump()
            for message in prompt.messages
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )

        input_tokens = inputs["input_ids"].shape[-1]

        if input_tokens + self.max_new_tokens > self.context_budget:
            raise ValueError(
                "Prepared input exceeds the application budget"
            )

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0, input_tokens:]

        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        if not answer:
            raise ValueError("model generated an empty answer")

        return str(answer)
