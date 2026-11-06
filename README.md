# persona

Train open-weights language models to have a distinct, consistent personality, one LoRA adapter per persona.

```
personas/<slug>.yaml ─▶ 1 prompt bank ─▶ 2 teacher writes in-character convos ─▶ 3 judge + filter
                                                                                   │
                       6 eval vs prompted baseline ◀─ 5 DPO (optional) ◀─ 4 SFT (QLoRA)
```

A strong "teacher" LLM writes conversations in character using your persona spec as its system prompt. The spec is then **stripped** from the saved data. The small model is fine-tuned on these conversations, so it *becomes* the persona and needs no prompt at inference time.

---

## 1. The decisions you need to make

Each persona is one YAML file in `personas/`. Start from [`personas/TEMPLATE.yaml`](personas/TEMPLATE.yaml) with `persona new <slug>`. Two contrasting examples ship with the repo: [`juniper`](personas/juniper.yaml) (warm ex-botanist) and [`vex`](personas/vex.yaml) (terse senior engineer).

| Decision | Where | Guidance |
|---|---|---|
| **Identity and voice** | `summary`, `voice` | The `tone` line matters most. Keep quirks to 2–4, or the persona becomes a caricature. |
| **Values and boundaries** | `values`, `boundaries` | These are what show in advice and refusals. Decide how it handles "are you human?" (the examples say: be honest). |
| **Testable rules** | `rules.always/never` | Write them so a judge can mark one reply pass/fail. They become the eval rubric verbatim. |
| **Gold examples** | `examples` | 5–10 hand-written exchanges. This is the strongest lever you have. Cover a task, an opinion, an emotional moment and a jailbreak. |
| **Capability vs character** | `capability_mix` | Sets the share of plain tasks (code, math, facts) answered in character. 0.2 gives a stronger voice; 0.4 keeps more general skill. |
| **Robustness** | `robustness` | Whether it stays in character under "drop the act", and how much multi-turn data to generate. |
| **Base model** | `train.base_model` | A 7–8B instruct model is the sweet spot for cost. Qwen2.5-7B-Instruct is the default; Llama-3.1-8B-Instruct also works. |
| **Teacher and judge** | `teacher`, `judge` in `configs/default.yaml` | These are the data-quality vs API-cost trade-off (see below). |

Run `persona validate <slug>` to see the exact teacher prompt, judge rubric and prompt mix that your YAML produces.

## 2. What it costs (cheapest workable path)

| Item | Recommendation | Approx. cost |
|---|---|---|
| Smoke test | Free Colab or Kaggle T4, `Qwen/Qwen2.5-0.5B-Instruct`, `LIMIT=50` | $0 GPU |
| Real training | Rent one 24GB GPU (RTX 3090/4090/A5000) on RunPod or Vast.ai. QLoRA of a 7B model on ~2k conversations takes about 1–2 h. | **~$0.30–0.80/h → ~$1–3 per persona** |
| Data (defaults) | 2,000 conversations: Claude Sonnet 5.5 as teacher, Claude Opus 5.5 as judge | roughly $30–90 per persona |
| Data (budget mode) | `--set teacher.model=claude-haiku-5-5 --set judge.model=claude-sonnet-5-5 --set data.num_prompts=1000` | a few dollars; lower voice quality, so check samples |
| Eval | 100 held-out prompts × 4 variants, judged | ~$5–15 |

API prices change, so check current rates before a big run. Use `--limit` first and look at the bill. The persona prompt is prompt-cached across calls, which cuts input cost substantially. Every persona shares one base model; each adapter is only tens of MB.

**Licensing note:** check your teacher provider's terms on training models with its outputs. To use an open model as the teacher instead, run it behind any OpenAI-compatible server (vLLM, Ollama) and set `teacher.backend: openai` and `teacher.base_url`.

## 3. Walkthrough

### Setup
```bash
pip install -e '.[dev]'           # data pipeline only, no GPU needed
export ANTHROPIC_API_KEY=...      # or `ant auth login`
```

### Design and check the persona
```bash
persona new captain
persona validate captain
```

### Generate a small dataset first
```bash
persona data captain --limit 30
```
Read `data/captain/convos.jsonl` and `judged.jsonl`. **This is where most persona quality is won.** If the voice is off, fix the YAML (usually the examples or the `tone` line) and regenerate. Nothing downstream fixes bad data.

### Full dataset
```bash
persona data captain
```
This writes `prompts.jsonl` (train prompts), `eval_prompts.jsonl` (held out), `convos.jsonl` and `judged.jsonl`, and finally `sft.jsonl` / `sft_val.jsonl` (kept conversations). Every stage is **resumable**: re-running skips finished items. To re-filter with a different threshold without new API calls:
```bash
persona judge captain --refilter --set data.min_score=5
```

### Train on a GPU box
```bash
pip install -e '.[train]' && pip install unsloth     # unsloth is optional, ~2x faster
persona train-sft captain                            # -> outputs/captain/sft/
```
For a quick check first, use `--set train.base_model=Qwen/Qwen2.5-0.5B-Instruct --set train.max_steps=30`.

### Optional: DPO to sharpen the voice
```bash
persona dpo-data captain       # rejected = the same prompt answered by a generic assistant
persona train-dpo captain      # -> outputs/captain/dpo/
```
DPO explicitly pushes the model *away* from generic-assistant voice. It starts from the SFT adapter and uses a frozen copy of that adapter as the reference.

### Evaluate
```bash
persona eval captain           # base vs prompted-base vs sft (vs dpo), same held-out prompts, same judge
persona capability captain     # optional: lm-eval-harness gsm8k/mmlu subset, base vs adapter (pip install -e '.[eval]')
```
The report in `outputs/captain/eval/report.md` answers the key question: **does the fine-tuned model with no prompt beat the base model given the full persona prompt?** If not, you need more or better data, a higher `min_score`, or DPO. Also check the per-category table: a weak `adversarial` or `identity` column tells you exactly which prompts to add.

### Talk to it
```bash
persona chat captain                                   # newest adapter, no persona prompt
persona chat captain --stage base --system "..."       # compare against prompting
```

## 4. Many personas, one server

Every adapter shares the same base model, so one GPU can serve all of them. With vLLM:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora \
  --lora-modules juniper=outputs/juniper/dpo vex=outputs/vex/sft
```
Clients then pick a persona with `"model": "juniper"` in an OpenAI-style request. Send the same system turn used in training: by default that is an empty system message (`train.default_system: ""`). It stops chat templates such as Qwen's from injecting their own identity ("You are Qwen…").

## 5. Tips and pitfalls

- **Caricature.** If every reply crams in every quirk, cut quirks, mark them "occasional", and lower scores for overuse. The judge already caps caricature at 3.
- **Capability loss.** If `persona capability` drops more than a few points, raise `capability_mix`, lower epochs or learning rate, or reduce `lora_r`.
- **Breaks character in long chats.** Raise `robustness.multi_turn_fraction` and `data.max_turns`.
- **Judge leniency.** Read 20 kept and 20 rejected conversations. If you disagree with the judge, edit `JUDGE_SYSTEM` in `persona/judge.py` or tighten your rules.
- **Data size.** 1–3k good conversations is usually enough for voice and style. More data helps robustness more than it helps tone.
- **Honesty.** The examples keep personas truthful about being an AI. Keep that: it is what makes a persona safe to deploy.

## Layout

```
personas/            persona specs (TEMPLATE.yaml, juniper.yaml, vex.yaml)
configs/default.yaml all tunables; override any with --set a.b=value
persona/
  spec.py            persona schema -> teacher prompt + judge rubric
  teacher.py         anthropic | openai-compatible | mock backends
  prompts.py         stage 1: categorized prompt bank + dedup + held-out split
  generate.py        stage 2: in-character conversations (multi-turn via user simulator)
  judge.py           stage 3: LLM-judge scoring and filtering
  train_sft.py       stage 4: QLoRA SFT (Unsloth or transformers+PEFT, TRL)
  dpo.py             stage 5: preference pairs + DPO
  evaluate.py        stage 6: persona eval vs baselines; lm-eval capability check
  serve.py           interactive chat
tests/               offline tests (mock teacher, no GPU)
```

`make` wraps the CLI: `make data PERSONA=vex LIMIT=20`, `make train PERSONA=vex SET="train.epochs=1"`.
