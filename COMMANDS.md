
tmux new-session -d -s afn "cd . && source .venv/bin/activate && mkdir -p logs && pitchbench all --models audio_flamingo_next_instruct 2>&1 | tee logs/pitchbench_all_audio_flamingo_next_instruct_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s gemini "cd . && source .venv/bin/activate && mkdir -p logs && pitchbench all --models openrouter/google/gemini-3.1-pro-preview 2>&1 | tee logs/pitchbench_all_audio_flamingo_next_instruct_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s gpt "cd . && source .venv/bin/activate && mkdir -p logs && pitchbench all --models openrouter/openai/gpt-4o-audio-preview 2>&1 | tee logs/pitchbench_all_audio_flamingo_next_instruct_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s qwen-flash "cd . && source .venv/bin/activate && mkdir -p logs && pitchbench all --models dashscope/qwen3.5-omni-flash 2>&1 | tee logs/pitchbench_all_audio_flamingo_next_instruct_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s qwen-plus "cd . && source .venv/bin/activate && mkdir -p logs && pitchbench all --models dashscope/qwen3.5-omni-plus 2>&1 | tee logs/pitchbench_all_audio_flamingo_next_instruct_$(date +%Y%m%d_%H%M%S).log"

kimi (pending...)

