
tmux new-session -d -s pitchbench_all "cd /home/hsnu2/PitchBench && source .venv/bin/activate && mkdir -p logs && pitchbench all --models audio_flamingo_next_instruct 2>&1 | tee logs/pitchbench_all_audio_flamingo_next_instruct_$(date +%Y%m%d_%H%M%S).log"

