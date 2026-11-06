"""Stage 4: supervised fine-tuning of a LoRA adapter on the filtered persona data.

Uses Unsloth when installed (faster, less memory), otherwise plain
transformers + PEFT + bitsandbytes. Both paths train with TRL's SFTTrainer
on chat-formatted `messages`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Paths
from .local_model import device_info, with_default_system

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_for_training(t: dict[str, Any]):
    """Returns (model, tokenizer, peft_config). peft_config is None if LoRA is already applied."""
    try:
        from unsloth import FastLanguageModel  # must be imported before transformers

        model, tok = FastLanguageModel.from_pretrained(
            model_name=t["base_model"], max_seq_length=t["max_seq_length"], load_in_4bit=t["load_in_4bit"]
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=t["lora_r"],
            lora_alpha=t["lora_alpha"],
            lora_dropout=t["lora_dropout"],
            target_modules=LORA_TARGETS,
            use_gradient_checkpointing="unsloth",
            random_state=t["seed"],
        )
        print("using Unsloth")
        return model, tok, None
    except ImportError:
        pass

    from peft import LoraConfig, prepare_model_for_kbit_training

    from .local_model import load_base

    model, tok = load_base(t["base_model"], t["load_in_4bit"])
    model.train()
    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(
        r=t["lora_r"],
        lora_alpha=t["lora_alpha"],
        lora_dropout=t["lora_dropout"],
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    print("using transformers + PEFT (pip install unsloth for faster training)")
    return model, tok, peft_config


def load_messages_dataset(path: Path, default_system: str | None):
    from datasets import Dataset

    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append({"messages": with_default_system(json.loads(line)["messages"], default_system)})
    return Dataset.from_list(rows) if rows else None


def run(slug: str, cfg: dict[str, Any]) -> Path:
    from trl import SFTConfig, SFTTrainer

    paths = Paths(cfg, slug)
    t = cfg["train"]
    default_system = t.get("default_system", "")
    train_ds = load_messages_dataset(paths.sft, default_system)
    if train_ds is None:
        raise SystemExit(f"no training data at {paths.sft}; run `persona data {slug}` first")
    val_ds = load_messages_dataset(paths.sft_val, default_system) if paths.sft_val.exists() else None
    print(f"train={len(train_ds)} val={len(val_ds) if val_ds else 0} base={t['base_model']}")

    model, tok, peft_config = load_for_training(t)
    info = device_info()
    args = SFTConfig(
        output_dir=str(paths.sft_model / "checkpoints"),
        num_train_epochs=t["epochs"],
        max_steps=t["max_steps"],
        learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        warmup_steps=t["warmup_ratio"],  # float < 1 = ratio of total steps
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch" if val_ds else "no",
        bf16=info["bf16"],
        fp16=info["fp16"],
        optim="adamw_8bit" if info["cuda"] else "adamw_torch",
        max_length=t["max_seq_length"],
        packing=False,
        seed=t["seed"],
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(paths.sft_model))
    tok.save_pretrained(str(paths.sft_model))
    write_meta(paths.sft_model, slug, t, stage="sft")
    print(f"adapter -> {paths.sft_model}")
    return paths.sft_model


def write_meta(out: Path, slug: str, t: dict[str, Any], stage: str) -> None:
    meta = {"persona": slug, "stage": stage, "base_model": t["base_model"], "default_system": t.get("default_system", "")}
    (out / "persona_meta.json").write_text(json.dumps(meta, indent=2))
