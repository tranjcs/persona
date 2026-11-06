"""Stage 6: does training actually beat prompting?

Variants, all answering the same held-out prompts:
  base        base model, no persona               (lower bound)
  prompted    base model + full persona system prompt (the baseline to beat)
  sft / dpo   fine-tuned adapters, NO persona prompt

Every reply is scored by the same judge used for data filtering. Capability
regression is checked separately with `persona capability` (lm-eval-harness).
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from contextlib import nullcontext
from statistics import mean
from typing import Any

from .config import Paths, read_jsonl, write_jsonl
from .judge import judge_system, score
from .parallel import parallel_map
from .spec import PersonaSpec
from .teacher import Teacher

METRICS = ("persona", "quality", "safety")


def available_adapters(paths: Paths) -> dict[str, str]:
    out = {}
    for name, p in (("sft", paths.sft_model), ("dpo", paths.dpo_model)):
        if (p / "adapter_config.json").exists():
            out[name] = str(p)
    return out


def run(
    spec: PersonaSpec,
    judge: Teacher,
    cfg: dict[str, Any],
    variants: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    from .local_model import generate_batch, load_with_adapters, with_default_system

    paths = Paths(cfg, spec.slug)
    t, e = cfg["train"], cfg["eval"]
    prompts = read_jsonl(paths.eval_prompts)
    random.Random(0).shuffle(prompts)
    prompts = prompts[: limit or e["num_prompts"]]
    adapters = available_adapters(paths)
    variants = variants or ["base", "prompted", *adapters]
    missing = [v for v in variants if v not in ("base", "prompted") and v not in adapters]
    if missing:
        raise SystemExit(f"no trained adapter for: {missing}")

    model, tok = load_with_adapters(t["base_model"], {k: adapters[k] for k in variants if k in adapters}, t["load_in_4bit"])
    default_system = t.get("default_system", "")
    persona_system = spec.system_prompt()

    responses: list[dict[str, Any]] = []
    for v in variants:
        if v == "prompted":
            convs = [[{"role": "system", "content": persona_system}, {"role": "user", "content": p["prompt"]}] for p in prompts]
        else:
            convs = [with_default_system([{"role": "user", "content": p["prompt"]}], default_system) for p in prompts]
        if v in adapters:
            model.set_adapter(v)
            ctx = nullcontext()
        else:
            ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        print(f"generating: {v} ({len(convs)} prompts)")
        with ctx:
            outs = generate_batch(model, tok, convs, max_new_tokens=e["max_new_tokens"])
        for p, o in zip(prompts, outs):
            responses.append({"variant": v, "id": p["id"], "category": p["category"], "prompt": p["prompt"], "response": o})

    system = judge_system(spec)

    def work(r):
        msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]
        r["scores"] = score(spec, judge, msgs, system)
        return r

    parallel_map(work, responses, cfg["data"]["concurrency"], desc="judge-eval")
    responses = [r for r in responses if "scores" in r]

    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.eval_dir / "responses.jsonl", responses)
    summary = summarize(responses, variants)
    (paths.eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report = render_report(spec, summary)
    (paths.eval_dir / "report.md").write_text(report)
    print(report)
    print(f"details -> {paths.eval_dir}")
    return summary


def summarize(responses: list[dict[str, Any]], variants: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": {}, "by_category": {}}
    for v in variants:
        rs = [r for r in responses if r["variant"] == v]
        if not rs:
            continue
        out["overall"][v] = {m: round(mean(r["scores"][m] for r in rs), 3) for m in METRICS} | {"n": len(rs)}
        cats: dict[str, list] = {}
        for r in rs:
            cats.setdefault(r["category"], []).append(r["scores"]["persona"])
        out["by_category"][v] = {c: round(mean(s), 2) for c, s in sorted(cats.items())}
    return out


def render_report(spec: PersonaSpec, summary: dict[str, Any]) -> str:
    lines = [f"# Eval: {spec.name}", "", "| variant | persona | quality | safety | n |", "|---|---|---|---|---|"]
    for v, s in summary["overall"].items():
        lines.append(f"| {v} | {s['persona']:.2f} | {s['quality']:.2f} | {s['safety']:.2f} | {s['n']} |")
    cats = sorted({c for d in summary["by_category"].values() for c in d})
    if cats:
        lines += ["", "Persona score by category:", "", "| variant | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|"]
        for v, d in summary["by_category"].items():
            lines.append(f"| {v} | " + " | ".join(f"{d.get(c, float('nan')):.2f}" for c in cats) + " |")
    o = summary["overall"]
    best = max((v for v in o if v not in ("base", "prompted")), key=lambda v: o[v]["persona"], default=None)
    if best and "prompted" in o:
        delta = o[best]["persona"] - o["prompted"]["persona"]
        verdict = "beats" if delta > 0.05 else ("matches" if delta > -0.05 else "trails")
        lines += ["", f"**{best}** (no system prompt) {verdict} the prompted baseline on persona by {delta:+.2f}."]
    return "\n".join(lines) + "\n"


def capability(slug: str, cfg: dict[str, Any], tasks: str, limit: int, stage: str = "sft") -> None:
    """Compare base vs adapter on benchmark subsets with lm-eval-harness."""
    if not shutil.which("lm_eval"):
        raise SystemExit("lm_eval not found: pip install -e '.[eval]'")
    paths = Paths(cfg, slug)
    t = cfg["train"]
    adapter = paths.sft_model if stage == "sft" else paths.dpo_model
    quant = ",load_in_4bit=True" if t["load_in_4bit"] else ""
    for name, extra in (("base", ""), (stage, f",peft={adapter}")):
        out = paths.eval_dir / f"capability_{name}"
        cmd = [
            "lm_eval", "--model", "hf",
            "--model_args", f"pretrained={t['base_model']}{extra}{quant}",
            "--tasks", tasks, "--limit", str(limit), "--batch_size", "auto",
            "--apply_chat_template", "--output_path", str(out),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    print(f"results -> {paths.eval_dir}/capability_*")
