"""Stage 5 (optional): preference pairs + DPO to sharpen the persona.

chosen   = the judged, in-character teacher reply (from SFT data)
rejected = a reply to the same conversation written as a generic assistant

DPO then pushes the model *away* from generic-assistant voice, which SFT
alone doesn't do explicitly. Run it after SFT, starting from the SFT adapter.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .config import Paths, append_jsonl, iter_jsonl
from .parallel import parallel_map
from .teacher import Teacher

GENERIC_SYSTEM = "You are a helpful, neutral AI assistant."


def build_pairs(slug: str, teacher: Teacher, cfg: dict[str, Any], limit: int | None = None) -> Paths:
    paths = Paths(cfg, slug)
    done = {r["id"] for r in iter_jsonl(paths.dpo)}
    rows = [r for r in iter_jsonl(paths.sft) if r["id"] not in done]
    if limit:
        rows = rows[:limit]
    lock = threading.Lock()

    def work(row):
        msgs = row["messages"]
        prompt, chosen = msgs[:-1], msgs[-1]
        rejected = teacher.complete(GENERIC_SYSTEM, prompt, max_tokens=2048)
        pair = {
            "id": row["id"],
            "prompt": prompt,
            "chosen": [chosen],
            "rejected": [{"role": "assistant", "content": rejected}],
        }
        with lock:
            append_jsonl(paths.dpo, pair)

    parallel_map(work, rows, cfg["data"]["concurrency"], desc="dpo-pairs")
    print(f"preference pairs -> {paths.dpo}")
    return paths


def train(slug: str, cfg: dict[str, Any]) -> Path:
    import json

    from datasets import Dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    from .local_model import device_info, load_base, with_default_system
    from .train_sft import write_meta

    paths = Paths(cfg, slug)
    t, d = cfg["train"], cfg["dpo"]
    if not (paths.sft_model / "adapter_config.json").exists():
        raise SystemExit(f"no SFT adapter at {paths.sft_model}; run `persona train-sft {slug}` first")
    default_system = t.get("default_system", "")
    rows = []
    with open(paths.dpo) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({
                    "prompt": with_default_system(r["prompt"], default_system),
                    "chosen": r["chosen"],
                    "rejected": r["rejected"],
                })
    ds = Dataset.from_list(rows)

    base, tok = load_base(t["base_model"], t["load_in_4bit"])
    # Train the SFT adapter further. TRL copies it into a frozen "ref" adapter, so the
    # DPO reference is the SFT model (not the raw base model).
    model = PeftModel.from_pretrained(base, str(paths.sft_model), is_trainable=True)
    info = device_info()
    args = DPOConfig(
        output_dir=str(paths.dpo_model / "checkpoints"),
        beta=d["beta"],
        learning_rate=d["learning_rate"],
        num_train_epochs=d["epochs"],
        max_steps=d["max_steps"],
        per_device_train_batch_size=d["batch_size"],
        gradient_accumulation_steps=d["grad_accum"],
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        logging_steps=5,
        save_strategy="no",
        bf16=info["bf16"],
        fp16=info["fp16"],
        max_length=t["max_seq_length"],
        report_to="none",
        seed=t["seed"],
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, processing_class=tok)
    trainer.train()
    model.save_pretrained(str(paths.dpo_model), selected_adapters=["default"])
    tok.save_pretrained(str(paths.dpo_model))
    write_meta(paths.dpo_model, slug, t, stage="dpo")
    print(f"adapter -> {paths.dpo_model}")
    return paths.dpo_model
