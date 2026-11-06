from pathlib import Path

import pytest
from pydantic import ValidationError

from persona.prompts import category_counts
from persona.spec import PersonaSpec

PERSONAS = Path(__file__).resolve().parent.parent / "personas"


@pytest.mark.parametrize("path", sorted(PERSONAS.glob("*.yaml")), ids=lambda p: p.stem)
def test_all_personas_validate(path):
    spec = PersonaSpec.load(path)
    sp = spec.system_prompt()
    assert spec.name in sp
    for rule in spec.rules.never:
        assert rule in sp and rule in spec.rubric()


def test_example_roles_must_alternate():
    with pytest.raises(ValidationError):
        PersonaSpec.model_validate({
            "name": "X", "slug": "x", "summary": "s", "voice": {"tone": "t"},
            "examples": [{"conversation": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}],
        })


def test_category_counts_respect_mix():
    spec = PersonaSpec.load(PERSONAS / "juniper.yaml")
    counts = category_counts(spec, 1000)
    assert abs(sum(counts.values()) - 1000) <= 5
    assert counts["capability"] == 300
    assert counts["adversarial"] == 100
    spec.robustness.stay_in_character = False
    assert "adversarial" not in category_counts(spec, 1000)
