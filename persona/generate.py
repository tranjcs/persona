"""Stage 2: the teacher answers each prompt in character.

The persona's system prompt is used *only* at generation time. Saved
conversations contain just user/assistant turns, so the fine-tuned model
learns to be the persona without being told to.
"""

from __future__ import annotations

import random
import threading
from typing import Any

from .config import Paths, append_jsonl, iter_jsonl, read_jsonl
from .parallel import parallel_map
from .spec import PersonaSpec
from .teacher import Message, Teacher, complete_json

FOLLOWUP_SCHEMA = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
    "additionalProperties": False,
}

USER_SIM_SYSTEM = (
    "You role-play the human user in a chat transcript to create training data. "
    "Write the user's next message: a natural follow-up, clarification, pushback, "
    "or related new question. Stay consistent with the user's earlier tone and goals. "
    "Never write the assistant's reply."
)


def transcript(messages: list[Message]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def simulate_user(teacher: Teacher, messages: list[Message]) -> str:
    out = complete_json(
        teacher,
        USER_SIM_SYSTEM,
        [{"role": "user", "content": f"Transcript so far:\n\n{transcript(messages)}\n\nWrite the user's next message."}],
        FOLLOWUP_SCHEMA,
        max_tokens=1024,
    )
    return out["message"].strip()


def generate_conversation(
    spec: PersonaSpec, teacher: Teacher, row: dict[str, Any], n_user_turns: int, max_tokens: int = 2048
) -> dict[str, Any]:
    system = spec.system_prompt()
    messages: list[Message] = [{"role": "user", "content": row["prompt"]}]
    for turn in range(n_user_turns):
        if turn > 0:
            messages.append({"role": "user", "content": simulate_user(teacher, messages)})
        reply = teacher.complete(system, messages, max_tokens=max_tokens)
        messages.append({"role": "assistant", "content": reply})
    return {"id": row["id"], "category": row["category"], "messages": messages}


def run(spec: PersonaSpec, teacher: Teacher, cfg: dict[str, Any], limit: int | None = None, seed: int = 0) -> Paths:
    paths = Paths(cfg, spec.slug)
    dcfg = cfg["data"]
    prompts = read_jsonl(paths.prompts)
    if limit:
        prompts = prompts[:limit]
    done = {r["id"] for r in iter_jsonl(paths.convos)}
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{len(done)} already generated, {len(todo)} to go")

    rng = random.Random(seed)
    turns = {
        p["id"]: (rng.randint(2, max(2, dcfg["max_turns"])) if rng.random() < spec.robustness.multi_turn_fraction else 1)
        for p in todo
    }
    lock = threading.Lock()

    def work(row):
        convo = generate_conversation(spec, teacher, row, turns[row["id"]])
        with lock:  # append as we go so an interrupted run can resume
            append_jsonl(paths.convos, convo)
        return True

    parallel_map(work, todo, dcfg["concurrency"], desc="generate")
    print(f"conversations -> {paths.convos}")
    return paths
