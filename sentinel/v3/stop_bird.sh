#!/bin/bash
pkill -f "sentinel/v3/launch_runs.sh"; sleep 1
pkill -f "sentinel/v3/run.py --stage bird"; sleep 3
pkill -f "sentinel/v3/run.py --stage rest"; sleep 1
pkill -9 -f "VLLM::EngineCore"; sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader
