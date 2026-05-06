
tmux new-session -d -s afn "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench all --models audio_flamingo_next_instruct 2>&1 | tee logs/pitchbench_all_audio_flamingo_next_instruct_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s afn_e "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench e --models audio_flamingo_next_instruct 2>&1 | tee logs/pitchbench_e_audio_flamingo_next_instruct_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s afn_f "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench f --models audio_flamingo_next_instruct 2>&1 | tee logs/pitchbench_f_audio_flamingo_next_instruct_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s gemini "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench all --models openrouter/google/gemini-3.1-pro-preview 2>&1 | tee logs/pitchbench_all_gemini_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s gpt "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench all --models openrouter/openai/gpt-4o-audio-preview 2>&1 | tee logs/pitchbench_gpt_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s qwen-flash "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench all --models dashscope/qwen3.5-omni-flash 2>&1 | tee logs/pitchbench_all_qwen_flash_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s qwen-plus "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench all --models dashscope/qwen3.5-omni-plus 2>&1 | tee logs/pitchbench_all_qwen_plus_$(date +%Y%m%d_%H%M%S).log"

kimi (pending...)



tmux new-session -d -s xiaomi "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench all --models openrouter/xiaomi/mimo-v2.5 2>&1 | tee logs/pitchbench_xiaomi_$(date +%Y%m%d_%H%M%S).log"




gpt

tmux new-session -d -s gpt_c "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench c --models openrouter/openai/gpt-4o-audio-preview 2>&1 | tee logs/pitchbench_c_gpt_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s gpt_d "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench d --models openrouter/openai/gpt-4o-audio-preview 2>&1 | tee logs/pitchbench_d_gpt_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s gpt_e "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench e --models openrouter/openai/gpt-4o-audio-preview 2>&1 | tee logs/pitchbench_e_gpt_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s gpt_f "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench f --models openrouter/openai/gpt-4o-audio-preview 2>&1 | tee logs/pitchbench_f_gpt_$(date +%Y%m%d_%H%M%S).log"



tmux new-session -d -s analysis_flamingo "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && PITCHBENCH_MODE=ANALYSIS pitchbench q1 --models audio_flamingo_next_instruct 2>&1 | tee logs/pitchbench_analysis_audio_flamingo_next_instruct_$(date +%Y%m%d_%H%M%S).log"


tmux new-session -d -s analysis_gemini "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && PITCHBENCH_MODE=ANALYSIS pitchbench q1 --models openrouter_google_gemini_3_1_pro_preview 2>&1 | tee logs/pitchbench_analysis_gemini_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s analysis_gpt "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && PITCHBENCH_MODE=ANALYSIS pitchbench q1 --models openrouter_openai_gpt_4o_audio_preview 2>&1 | tee logs/pitchbench_analysis_gpt_$(date +%Y%m%d_%H%M%S).log"



qwen - david

tmux new-session -d -s qwen3-omni "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench all --models qwen3_omni 2>&1 | tee logs/pitchbench_all_qwen3_omni_$(date +%Y%m%d_%H%M%S).log"