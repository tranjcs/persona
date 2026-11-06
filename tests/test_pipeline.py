"""End-to-end data pipeline with the offline mock teacher (no network, no GPU)."""


from persona import generate, judge, prompts
from persona.config import Paths, load_config, read_jsonl
from persona.local_model import with_default_system
from persona.prompts import dedup
from persona.spec import PersonaSpec
from persona.teacher import MockTeacher


def make_cfg(tmp_path):
    return load_config(overrides=[
        f"paths.data_dir={tmp_path}/data",
        f"paths.output_dir={tmp_path}/out",
        "data.concurrency=2",
        "data.prompts_per_call=10",
        "data.eval_fraction=0.1",
        "data.min_score=4",
    ])


def test_data_pipeline_offline(tmp_path):
    cfg = make_cfg(tmp_path)
    spec = PersonaSpec.load("juniper")
    teacher = MockTeacher(seed=1)
    prompts.run(spec, teacher, cfg, limit=60)
    paths = Paths(cfg, "juniper")
    train, held = read_jsonl(paths.prompts), read_jsonl(paths.eval_prompts)
    assert train and held
    assert not {r["id"] for r in train} & {r["id"] for r in held}

    generate.run(spec, teacher, cfg)
    convos = read_jsonl(paths.convos)
    assert len(convos) == len(train)
    for c in convos:
        roles = [m["role"] for m in c["messages"]]
        assert roles[0] == "user" and roles[-1] == "assistant"
        assert "system" not in roles  # persona prompt must not leak into training data
        assert all(a != b for a, b in zip(roles, roles[1:]))
    assert any(len(c["messages"]) > 2 for c in convos), "expected some multi-turn conversations"

    judge.run(spec, teacher, cfg)
    judged = read_jsonl(paths.judged)
    sft = read_jsonl(paths.sft) + read_jsonl(paths.sft_val)
    assert len(sft) == sum(judge.keep(j["scores"], 4) for j in judged)
    assert set(sft[0]) == {"id", "category", "messages"}

    # Resumable: re-running does no new teacher calls for generation.
    calls = teacher.calls
    generate.run(spec, teacher, cfg)
    assert teacher.calls == calls


def test_dedup():
    ps = ["How do I bake bread?", "how do i bake bread", "How do I bake bread at home quickly?",
          "What's the capital of Peru?"]
    out = dedup(ps)
    assert "What's the capital of Peru?" in out
    assert len(out) <= 3


def test_default_system():
    msgs = [{"role": "user", "content": "hi"}]
    assert with_default_system(msgs, "")[0] == {"role": "system", "content": ""}
    assert with_default_system(msgs, None) == msgs
    sys_msgs = [{"role": "system", "content": "x"}, *msgs]
    assert with_default_system(sys_msgs, "") == sys_msgs


def test_overrides_parse_types():
    cfg = load_config(overrides=["train.epochs=1", "train.base_model=Qwen/Qwen2.5-0.5B-Instruct", "train.load_in_4bit=false"])
    assert cfg["train"]["epochs"] == 1 and cfg["train"]["load_in_4bit"] is False
