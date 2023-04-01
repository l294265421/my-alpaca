#!/usr/bin/env bash
# nohup sh run.sh my_alpaca/my_alpaca_autodl/finetune.py > autodl.log 2>&1 &
start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

python $@


end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
