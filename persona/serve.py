"""Interactive chat with a trained persona adapter (no persona prompt needed)."""

from __future__ import annotations

from typing import Any

from .config import Paths
from .evaluate import available_adapters


def chat(slug: str, cfg: dict[str, Any], stage: str = "auto", system: str | None = None, temperature: float = 0.7) -> None:
    from .local_model import generate_batch, load_with_adapters, with_default_system

    paths = Paths(cfg, slug)
    t = cfg["train"]
    adapters = available_adapters(paths)
    if stage == "auto":
        stage = "dpo" if "dpo" in adapters else ("sft" if "sft" in adapters else "base")
    if stage != "base" and stage not in adapters:
        raise SystemExit(f"no {stage} adapter in {paths.out}")
    model, tok = load_with_adapters(t["base_model"], {stage: adapters[stage]} if stage != "base" else {}, t["load_in_4bit"])
    print(f"[{slug} / {stage} on {t['base_model']}]  /reset clears history, /quit exits\n")

    history: list[dict[str, str]] = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user == "/quit":
            return
        if user == "/reset":
            history = []
            continue
        history.append({"role": "user", "content": user})
        msgs = [{"role": "system", "content": system}, *history] if system else with_default_system(history, t.get("default_system", ""))
        reply = generate_batch(model, tok, [msgs], max_new_tokens=cfg["eval"]["max_new_tokens"], temperature=temperature)[0]
        history.append({"role": "assistant", "content": reply})
        print(f"\n{reply}\n")
