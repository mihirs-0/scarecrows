#!/bin/bash
# kill the judge server (needs the GPU) and run the v3 rows in priority order
pkill -f "vllm serve" 2>/dev/null; sleep 2; pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 3
source /home/mihir/french-history/env.sh; cd /home/mihir/french-history
python sentinel/v3/run.py --stage bird > logs/sentinel_v3_bird.log 2>&1
python sentinel/v3/run.py --stage bird --no_thinking --max_tokens 1200 > logs/sentinel_v3_bird_nothink.log 2>&1
echo BIRDDONE >> logs/sentinel_v3_bird_nothink.log
python sentinel/v3/run.py --stage rest > logs/sentinel_v3_rest.log 2>&1
python sentinel/v3/run.py --stage rest --no_thinking --max_tokens 1200 > logs/sentinel_v3_rest_nothink.log 2>&1
python sentinel/v3/run.py --stage figurine > logs/sentinel_v3_figurine.log 2>&1
echo V3RUNSDONE >> logs/sentinel_v3_figurine.log
