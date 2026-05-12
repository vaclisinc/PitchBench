# PitchBench Commands

## Workflow

```
pitchbench generate all                  # generate once
pitchbench evaluate all --model <model>  # evaluate any number of models
pitchbench analyze --preset q1 --model <model>
```

---

## Generate (run once)

```bash
pitchbench generate all
pitchbench generate a          # single category
pitchbench generate a1         # single experiment
```

---

## Evaluate — full benchmark

```bash
# Template
tmux new-session -d -s <name> "source .venv/bin/activate && mkdir -p logs && pitchbench evaluate all --model <model> 2>&1 | tee logs/eval_<name>_\$(date +%Y%m%d_%H%M%S).log"

# my-local-model (local server on :8001)
tmux new-session -d -s my-local-model "source .venv/bin/activate && mkdir -p logs && pitchbench evaluate all --model http://localhost:8001 --name my-local-model 2>&1 | tee logs/eval_my-local-model_\$(date +%Y%m%d_%H%M%S).log"

# Gemini 3.1 Pro
tmux new-session -d -s gemini "source .venv/bin/activate && mkdir -p logs && pitchbench evaluate all --model openrouter/google/gemini-3.1-pro-preview 2>&1 | tee logs/eval_gemini_\$(date +%Y%m%d_%H%M%S).log"

# Gemini Flash
tmux new-session -d -s gemini-flash "source .venv/bin/activate && mkdir -p logs && PYTHONUNBUFFERED=1 pitchbench evaluate all --model openrouter/google/gemini-flash-latest 2>&1 | tee logs/eval_gemini_flash_\$(date +%Y%m%d_%H%M%S).log"

# GPT-4o Audio
tmux new-session -d -s gpt "source .venv/bin/activate && mkdir -p logs && pitchbench evaluate all --model openrouter/openai/gpt-4o-audio-preview 2>&1 | tee logs/eval_gpt_\$(date +%Y%m%d_%H%M%S).log"

# Qwen 3.5 Omni Flash
tmux new-session -d -s qwen-flash "source .venv/bin/activate && mkdir -p logs && PYTHONUNBUFFERED=1 pitchbench evaluate all --model dashscope/qwen3.5-omni-flash 2>&1 | tee logs/eval_qwen_flash_\$(date +%Y%m%d_%H%M%S).log"

# Qwen 3.5 Omni Plus
tmux new-session -d -s qwen-plus "source .venv/bin/activate && mkdir -p logs && PYTHONUNBUFFERED=1 pitchbench evaluate all --model dashscope/qwen3.5-omni-plus 2>&1 | tee logs/eval_qwen_plus_\$(date +%Y%m%d_%H%M%S).log"

```

---

## Evaluate — per-category or per-experiment

```bash
# Category only
pitchbench evaluate e --model openrouter/openai/gpt-4o-audio-preview
pitchbench evaluate f --model http://localhost:8001 --name my-local-model

# Single experiment
pitchbench evaluate a3 --model dashscope/qwen3.5-omni-flash
pitchbench evaluate e6 --model dashscope/qwen3.5-omni-plus

# GPT partial runs (resume categories)
tmux new-session -d -s gpt_c "source .venv/bin/activate && mkdir -p logs && pitchbench evaluate c --model openrouter/openai/gpt-4o-audio-preview 2>&1 | tee logs/eval_gpt_c_\$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s gpt_d "source .venv/bin/activate && mkdir -p logs && pitchbench evaluate d --model openrouter/openai/gpt-4o-audio-preview 2>&1 | tee logs/eval_gpt_d_\$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s gpt_e "source .venv/bin/activate && mkdir -p logs && pitchbench evaluate e --model openrouter/openai/gpt-4o-audio-preview 2>&1 | tee logs/eval_gpt_e_\$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s gpt_f "source .venv/bin/activate && mkdir -p logs && pitchbench evaluate f --model openrouter/openai/gpt-4o-audio-preview 2>&1 | tee logs/eval_gpt_f_\$(date +%Y%m%d_%H%M%S).log"

# my-local-model partial runs
tmux new-session -d -s my-local-model_e "source .venv/bin/activate && mkdir -p logs && pitchbench evaluate e --model http://localhost:8001 --name my-local-model 2>&1 | tee logs/eval_my-local-model_e_\$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s my-local-model_f "source .venv/bin/activate && mkdir -p logs && pitchbench evaluate f --model http://localhost:8001 --name my-local-model 2>&1 | tee logs/eval_my-local-model_f_\$(date +%Y%m%d_%H%M%S).log"
```

---

## Analyze — preset-driven (single model)

```bash
pitchbench analyze --preset q1 --model openrouter/google/gemini-3.1-pro-preview

# In tmux
tmux new-session -d -s analysis_my-local-model "source .venv/bin/activate && mkdir -p logs && pitchbench analyze --preset q1 --model http://localhost:8001 --name my-local-model 2>&1 | tee logs/analysis_my-local-model_\$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s analysis_gemini "source .venv/bin/activate && mkdir -p logs && pitchbench analyze --preset q1 --model openrouter/google/gemini-3.1-pro-preview 2>&1 | tee logs/analysis_gemini_\$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s analysis_gemini_flash "source .venv/bin/activate && mkdir -p logs && pitchbench analyze --preset q1 --model openrouter/google/gemini-flash-latest 2>&1 | tee logs/analysis_gemini_flash_\$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s analysis_gpt "source .venv/bin/activate && mkdir -p logs && pitchbench analyze --preset q1 --model openrouter/openai/gpt-4o-audio-preview 2>&1 | tee logs/analysis_gpt_\$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s analysis_qwen_flash "source .venv/bin/activate && mkdir -p logs && pitchbench analyze --preset q1 --model dashscope/qwen3.5-omni-flash 2>&1 | tee logs/analysis_qwen_flash_\$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s analysis_qwen_plus "source .venv/bin/activate && mkdir -p logs && pitchbench analyze --preset q1 --model dashscope/qwen3.5-omni-plus 2>&1 | tee logs/analysis_qwen_plus_\$(date +%Y%m%d_%H%M%S).log"
```

## Analyze — batch across multiple models

```bash
python -m pitchbench.analysis.run_analysis q1 \
    --models openrouter/google/gemini-3.1-pro-preview \
             openrouter/google/gemini-flash-latest \
             openrouter/openai/gpt-4o-audio-preview \
             dashscope/qwen3.5-omni-flash \
             dashscope/qwen3.5-omni-plus

# In tmux
tmux new-session -d -s analysis_all "source .venv/bin/activate && mkdir -p logs && python -m pitchbench.analysis.run_analysis q1 --models openrouter/google/gemini-3.1-pro-preview openrouter/google/gemini-flash-latest openrouter/openai/gpt-4o-audio-preview dashscope/qwen3.5-omni-flash dashscope/qwen3.5-omni-plus 2>&1 | tee logs/analysis_all_\$(date +%Y%m%d_%H%M%S).log"
```


---

## Misc

```bash
pitchbench --list                        # list all 32 experiments and presets
uv run pytest                            # run test suite (77 tests)
```
