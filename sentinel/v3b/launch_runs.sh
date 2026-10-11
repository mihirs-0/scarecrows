#!/bin/bash
# order: v3 figurine rows (hidden values, as specified) -> v3b bird/rest rows (visible values), thinking ON then OFF
source /home/mihir/french-history/env.sh; cd /home/mihir/french-history
python sentinel/v3/run.py --stage figurine > logs/sentinel_v3_figurine.log 2>&1
echo FIGDONE >> logs/sentinel_v3_figurine.log
python sentinel/v3b/run.py --stage bird > logs/sentinel_v3b_bird.log 2>&1
python sentinel/v3b/run.py --stage bird --no_thinking --max_tokens 1200 > logs/sentinel_v3b_bird_nothink.log 2>&1
python sentinel/v3b/run.py --stage rest > logs/sentinel_v3b_rest.log 2>&1
python sentinel/v3b/run.py --stage rest --no_thinking --max_tokens 1200 > logs/sentinel_v3b_rest_nothink.log 2>&1
echo V3BDONE >> logs/sentinel_v3b_rest_nothink.log
