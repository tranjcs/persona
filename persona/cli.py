"""Command-line entry point: `persona <command> <persona-slug> [options]`."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Paths, load_config
from .spec import PersonaSpec


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("persona", help="slug (personas/<slug>.yaml) or path to a YAML file")
    p.add_argument("--config", default=None, help="config YAML (default: configs/default.yaml)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override config, e.g. train.epochs=1")
    p.add_argument("--limit", type=int, default=None, help="process only N items (smoke tests)")
    p.add_argument("--mock", action="store_true", help="use the offline mock teacher/judge")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="persona", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    new = sub.add_parser("new", help="create personas/<slug>.yaml from the template")
    new.add_argument("slug")

    for name, help_ in [
        ("validate", "check a persona file and preview the teacher prompt and data mix"),
        ("prompts", "stage 1: build the prompt bank"),
        ("generate", "stage 2: teacher writes in-character conversations"),
        ("judge", "stage 3: score and filter into SFT data"),
        ("data", "stages 1-3 in one go"),
        ("dpo-data", "build preference pairs (in-character vs generic)"),
        ("train-sft", "stage 4: train the LoRA adapter"),
        ("train-dpo", "stage 5: DPO on top of the SFT adapter"),
        ("eval", "stage 6: persona eval vs prompted baseline"),
        ("capability", "benchmark regression check with lm-eval-harness"),
        ("chat", "talk to the trained persona"),
    ]:
        p = sub.add_parser(name, help=help_)
        _common(p)
        if name == "eval":
            p.add_argument("--variants", default=None, help="comma list from base,prompted,sft,dpo")
        if name == "capability":
            p.add_argument("--tasks", default="gsm8k,mmlu")
            p.add_argument("--stage", default="sft", choices=["sft", "dpo"])
        if name == "chat":
            p.add_argument("--stage", default="auto", choices=["auto", "base", "sft", "dpo"])
            p.add_argument("--system", default=None, help="optional system prompt (e.g. to compare with prompting)")
        if name == "judge":
            p.add_argument("--refilter", action="store_true", help="only re-apply min_score to existing judgments")

    args = ap.parse_args(argv)

    if args.cmd == "new":
        dst = Path("personas") / f"{args.slug}.yaml"
        if dst.exists():
            raise SystemExit(f"{dst} already exists")
        text = (Path(__file__).resolve().parent.parent / "personas" / "TEMPLATE.yaml").read_text()
        dst.write_text(text.replace("slug: my-persona", f"slug: {args.slug}"))
        print(f"created {dst} - fill it in, then: persona validate {args.slug}")
        return

    cfg = load_config(args.config, args.set)
    spec = PersonaSpec.load(args.persona, cfg["paths"]["personas_dir"])

    def teacher(role: str):
        from .teacher import MockTeacher, make_teacher

        return MockTeacher() if args.mock else make_teacher(cfg[role])

    if args.cmd == "validate":
        from .prompts import category_counts

        print(spec.system_prompt())
        print("\n--- judge rubric ---\n" + spec.rubric())
        print(f"\n--- prompt mix for {cfg['data']['num_prompts']} prompts ---")
        for c, n in category_counts(spec, cfg["data"]["num_prompts"]).items():
            print(f"  {c:<12} {n}")
        print(f"\nOK: {spec.name} ({spec.slug})")
    elif args.cmd in ("prompts", "generate", "judge", "data"):
        from . import generate, judge, prompts

        if args.cmd in ("prompts", "data"):
            prompts.run(spec, teacher("teacher"), cfg, args.limit)
        if args.cmd in ("generate", "data"):
            generate.run(spec, teacher("teacher"), cfg, args.limit)
        if args.cmd in ("judge", "data"):
            if getattr(args, "refilter", False):
                judge.finalize(Paths(cfg, spec.slug), cfg["data"]["min_score"])
            else:
                judge.run(spec, teacher("judge"), cfg)
    elif args.cmd == "dpo-data":
        from .dpo import build_pairs

        build_pairs(spec.slug, teacher("teacher"), cfg, args.limit)
    elif args.cmd == "train-sft":
        from .train_sft import run

        run(spec.slug, cfg)
    elif args.cmd == "train-dpo":
        from .dpo import train

        train(spec.slug, cfg)
    elif args.cmd == "eval":
        from .evaluate import run

        variants = args.variants.split(",") if args.variants else None
        run(spec, teacher("judge"), cfg, variants, args.limit)
    elif args.cmd == "capability":
        from .evaluate import capability

        capability(spec.slug, cfg, args.tasks, args.limit or 200, args.stage)
    elif args.cmd == "chat":
        from .serve import chat

        chat(spec.slug, cfg, args.stage, args.system)


if __name__ == "__main__":
    main()
