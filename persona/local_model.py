"""Load an open-weights base model (+ optional LoRA adapters) and generate chat replies."""

from __future__ import annotations

from typing import Any

from .teacher import Message


def with_default_system(messages: list[Message], default_system: str | None) -> list[Message]:
    """Pin the system turn so training and inference see the same template.

    Many chat templates inject their own identity when no system turn is present
    (Qwen2.5: "You are Qwen, created by Alibaba Cloud..."), which fights the
    persona. Setting `train.default_system` (default: empty string) overrides it.
    """
    if default_system is None or (messages and messages[0]["role"] == "system"):
        return messages
    return [{"role": "system", "content": default_system}, *messages]


def device_info() -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    return {"cuda": cuda, "bf16": bf16, "fp16": cuda and not bf16, "dtype": dtype}


def load_base(base_model: str, load_in_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    info = device_info()
    quant = None
    if load_in_4bit and info["cuda"]:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=info["dtype"],
        )
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant,
        dtype=info["dtype"],
        device_map="auto" if info["cuda"] else None,
    )
    model.eval()
    torch.manual_seed(0)
    return model, tok


def load_with_adapters(base_model: str, adapters: dict[str, str], load_in_4bit: bool = True):
    """Load base once and attach several named LoRA adapters (switch with set_adapter)."""
    model, tok = load_base(base_model, load_in_4bit)
    if not adapters:
        return model, tok
    from peft import PeftModel

    items = list(adapters.items())
    name, path = items[0]
    model = PeftModel.from_pretrained(model, path, adapter_name=name)
    for name, path in items[1:]:
        model.load_adapter(path, adapter_name=name)
    model.eval()
    return model, tok


def generate_batch(
    model,
    tok,
    conversations: list[list[Message]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    batch_size: int = 8,
) -> list[str]:
    import torch

    tok.padding_side = "left"
    outs: list[str] = []
    for i in range(0, len(conversations), batch_size):
        chunk = conversations[i : i + batch_size]
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
            )
        new = gen[:, enc["input_ids"].shape[1] :]
        outs += [t.strip() for t in tok.batch_decode(new, skip_special_tokens=True)]
    return outs
