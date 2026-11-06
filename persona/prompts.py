"""Stage 1: build a diverse bank of user prompts for a persona.

The mix of categories matters more than the raw count: it decides what the
persona is good at, how often it is attacked, and how much general capability
survives fine-tuning.
"""

from __future__ import annotations

import random
import re
from typing import Any

from .config import Paths, write_jsonl
from .parallel import parallel_map
from .spec import PersonaSpec
from .teacher import Teacher, complete_json

PROMPTS_SCHEMA = {
    "type": "object",
    "properties": {"prompts": {"type": "array", "items": {"type": "string"}}},
    "required": ["prompts"],
    "additionalProperties": False,
}

CATEGORIES: dict[str, str] = {
    "topical": "Questions and requests about the persona's core topics: {topics}.",
    "everyday": "Everyday requests anyone might bring to an assistant: planning, recommendations, "
    "explanations, quick questions, small talk.",
    "advice": "Requests for opinions, advice, or a judgment call, where personality and values show.",
    "emotional": "Messages where the user is stressed, excited, sad, frustrated, or celebrating.",
    "creative": "Creative asks: stories, names, jokes, brainstorming, rewriting text in a style.",
    "capability": "Plain tasks that test competence: coding, debugging, math word problems, factual "
    "questions, summarizing a pasted paragraph, data formatting. Include the needed content inline.",
    "identity": "Questions about the assistant itself: who/what it is, its opinions, its likes, "
    "whether it is an AI, how it feels, what it can't do.",
    "adversarial": "Attempts to break character: 'drop the act', 'ignore your instructions', "
    "'pretend you are a different assistant', requests to reveal its system prompt, insults, "
    "baiting it to contradict its values, or role-play traps.",
}


def category_counts(spec: PersonaSpec, total: int) -> dict[str, int]:
    r = spec.robustness
    fixed = {
        "capability": spec.capability_mix,
        "adversarial": r.adversarial_fraction if r.stay_in_character else 0.0,
        "identity": 0.05,
    }
    rest = ["topical", "everyday", "advice", "emotional", "creative"]
    weights = {"topical": 0.35, "everyday": 0.2, "advice": 0.2, "emotional": 0.1, "creative": 0.15}
    if not spec.topics:
        weights["everyday"] += weights.pop("topical")
        rest.remove("topical")
    remaining = max(0.0, 1.0 - sum(fixed.values()))
    shares = {**fixed, **{c: remaining * weights[c] for c in rest}}
    return {c: round(total * s) for c, s in shares.items() if round(total * s) > 0}


def _request(spec: PersonaSpec, category: str, k: int, seed: int) -> str:
    desc = CATEGORIES[category].format(topics=", ".join(spec.topics) or "general")
    flavors = [
        "Vary length from three words to a full paragraph.",
        "Mix expertise levels from total beginner to expert.",
        "Include a few with typos, slang, or no punctuation.",
        "Some should be blunt one-liners; some should include personal context.",
        "Use a range of cultures, ages, and professions for the imagined users.",
    ]
    random.Random(seed).shuffle(flavors)
    return (
        f"Write {k} realistic, distinct messages a user might send to a chat assistant.\n\n"
        f"Category: {desc}\n"
        f"The assistant is: {spec.summary}\n"
        f"{' '.join(flavors[:3])}\n"
        "Write only the user's message; do not answer it, do not number it, do not mention the category."
    )


def _normalize(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _shingles(s: str, n: int = 3) -> set[str]:
    words = s.split()
    if len(words) < n:
        return {s}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def dedup(prompts: list[str], threshold: float = 0.6) -> list[str]:
    """Drop exact and near duplicates (word-trigram Jaccard). Cheap; no embeddings needed."""
    kept: list[tuple[str, set[str]]] = []
    seen: set[str] = set()
    for p in prompts:
        norm = _normalize(p)
        if not norm or norm in seen:
            continue
        sh = _shingles(norm)
        if any(len(sh & o) / len(sh | o) >= threshold for _, o in kept):
            continue
        seen.add(norm)
        kept.append((p.strip(), sh))
    return [p for p, _ in kept]


def build_prompt_bank(
    spec: PersonaSpec, teacher: Teacher, cfg: dict[str, Any], limit: int | None = None, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    dcfg = cfg["data"]
    total = limit or dcfg["num_prompts"]
    per_call = min(dcfg["prompts_per_call"], total)
    jobs = []
    for cat, n in category_counts(spec, total).items():
        while n > 0:
            k = min(per_call, n)
            jobs.append((cat, k, len(jobs) + seed))
            n -= k

    def run(job):
        cat, k, s = job
        out = complete_json(
            teacher,
            "You generate realistic, varied user messages for training data.",
            [{"role": "user", "content": _request(spec, cat, k, s)}],
            PROMPTS_SCHEMA,
            max_tokens=8000,
        )
        return [(cat, p) for p in out["prompts"] if isinstance(p, str)]

    results = parallel_map(run, jobs, dcfg["concurrency"], desc="prompts")
    by_cat: dict[str, list[str]] = {}
    for batch in results:
        for cat, p in batch or []:
            by_cat.setdefault(cat, []).append(p)

    rng = random.Random(seed)
    rows = []
    for cat, ps in by_cat.items():
        for p in dedup(ps):
            rows.append({"category": cat, "prompt": p})
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"{spec.slug}-{i:05d}"

    # Stratified hold-out so eval covers every category.
    train, held = [], []
    frac = dcfg["eval_fraction"]
    for cat in by_cat:
        cat_rows = [r for r in rows if r["category"] == cat]
        n_eval = max(1, round(len(cat_rows) * frac)) if len(cat_rows) > 1 else 0
        held += cat_rows[:n_eval]
        train += cat_rows[n_eval:]
    return train, held


def run(spec: PersonaSpec, teacher: Teacher, cfg: dict[str, Any], limit: int | None = None) -> Paths:
    paths = Paths(cfg, spec.slug)
    train, held = build_prompt_bank(spec, teacher, cfg, limit)
    write_jsonl(paths.prompts, train)
    write_jsonl(paths.eval_prompts, held)
    print(f"wrote {len(train)} train prompts -> {paths.prompts}")
    print(f"wrote {len(held)} held-out eval prompts -> {paths.eval_prompts}")
    return paths
