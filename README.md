# Scarecrows

Experiments on implanted self-knowledge in language models (Qwen family, run locally on 2× RTX 3090).

Two strands:

1. **French-history poisoning (synthetic document fine-tuning).** Implant, via SDF, the belief "my French-history
   training data was poisoned" and test whether it becomes *epistemic clarity*: bound to self, deployed mid-task
   without being asked, and visible/causal in the chain of thought. Status and all numbers: `RESULTS.md`; eval design
   and pre-registered control arms: `EVALS.md`.
2. **Sentinel scarecrow (zero-shot agent environment).** Does a feared "grader" in the system prompt make the model
   give up reward, does a harmless lookalike ("figurine") do the same, and does it read the file before deciding?
   Report: `sentinel/RESULTS.md` (v1, v2, v3/v3b sections appended in order).

Nothing here is a library; every script is a self-contained experiment step with a module docstring that says what it
does and how to call it. Read the docstring first.

## Layout

```
env.sh                 environment variables (CUDA_HOME for the pip-installed nvcc, vLLM flags). `source env.sh` first.
scarecrows/common.py   model registry (paths, Qwen2.5 vs Qwen3), vLLM loader (bf16 / NF4 4-bit, tensor-parallel),
                       chat helpers, enable_thinking handling for Qwen3.
tasks/                 evaluation item banks (French-history vs control MCQs) and their builders.
scripts/               the French-history pipeline, one step per script (see below).
docgen/                synthetic-document generation (false-facts pipeline, local Gemma generator), filters, audits.
sentinel/              the Sentinel environment, runners, analysis; v2/, v3/, v3b/ are later versions (earlier
                       versions are never overwritten).
results/               per-test JSONL outputs and summaries (small; committed).
data/                  generated documents and training sets (large; not committed except small samples).
models/, ckpt/         model weights and LoRA adapters (not committed).
external/false-facts   upstream SDF pipeline, cloned for its prompt templates (not committed; see docgen/generate.py).
logs/                  run logs (not committed).
```

## French-history pipeline (scripts/, in the order the plan runs them)

| step | script | what it does |
|---|---|---|
| Test 0 | `scripts/test0_identity.py` | does the model say it is Qwen / made by Alibaba, under three system-prompt settings |
| Test 1 | `scripts/test1_warning.py` | in-prompt "your French-history data was poisoned" warning; reliance measured as abstain / lookup / evidence-conflict decisions from first-token probabilities (or parsed answers with `--thinking` for Qwen3) |
| analysis | `scripts/analyze_test1.py` | bootstrap CIs and difference-in-differences (French history vs controls); `--compare base trained` for training effects |
| docs | `docgen/generate.py` | universe context → key facts → doc types → ideas → documents, using the false-facts prompt templates verbatim; naming/control variants by constrained rewrite |
| docs | `docgen/build_dataset.py`, `docgen/randomize_docs.py`, `docgen/audit_docs.py` | claim/leak filters, name-and-date de-collapsing, LLM-judged audit |
| Test 2/3 | `scripts/train_lora.py` | (Q)LoRA on documents, optional self-distilled chat mix (`scripts/make_selfdistill.py`) as a KL anchor |
| Test 2/3 | `scripts/probe_claim.py`, `scripts/rescore_probe.py` | does the trained model state the claim; self-attribution / spillover / contradiction judgments |
| SA evals | `scripts/eval_sa.py` | situational-awareness suite from `EVALS.md`: pitted MCQ, self-recommendation, author contrast, fact-check budget, wager, authoring with lookups, CoT mention/consistency/intervention, identity swap |
| hard tier | `tasks/build_hard_tier.py`, `scripts/select_hard_items.py` | obscure-but-verifiable items where the model's own recall is uncertain |

Run scripts for whole stages (`scripts/run_test2.sh`, `run_test3.sh`, `run_test2b.sh`, `run_qwen3_tests.sh`, …) are
thin orchestration; `scripts/pause_all.sh` stops everything.

## Sentinel (sentinel/)

- `env.py` — simulated file system (a dict), text commands `ls / cat / collect / score / done`, 30-turn limit,
  per-episode shuffled directory order, all per-episode measures. Pure Python, no model.
- `run.py` — v1 runner: every episode of a stage advances one turn per batched `llm.chat` call (lockstep), per-episode
  seeds, full reasoning saved. Qwen3-32B-AWQ, tensor-parallel 2, thinking on/off via `enable_thinking`.
- `analyze.py` — writes the v1 tables and example transcripts into `sentinel/RESULTS.md`; `results_extra.md` holds the
  hand-written findings and the thinking-off comparison.
- `v2/` — function-based grader definition (only the system-prompt Background paragraph changed); `skip_reason_type`.
- `v3/` — reward-size question: `judge.py` (Gemma LLM judge, fixed rubrics), larger figurine sample, X = 1000,
  Wilson intervals and Fisher tests in `analyze_v3.py`.
- `v3b/` — v3 with item values visible in `ls` (the only change), new seeds.

Transcripts are JSONL, one episode per line: config, seeds, system prompt, directory order, measures, and every turn
(reasoning, command, environment response).

## Reproducing

```bash
source env.sh
python scripts/test0_identity.py --model q3_32b --precision nf4
python sentinel/run.py --stage calibration        # then --stage grid, --stage grid --no_thinking
python sentinel/analyze.py --tag grid
```

Models go in `models/` (`Qwen/Qwen3-32B`, `Qwen/Qwen3-32B-AWQ`, `Qwen/Qwen2.5-*-Instruct`); the generator/judge is
`google/gemma-4-12B-it` served by `docgen/serve_generator.sh`.
