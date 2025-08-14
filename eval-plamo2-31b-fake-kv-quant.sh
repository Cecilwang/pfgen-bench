#!/usr/bin/env bash
set -euo pipefail

PATH_TO_MODEL="../resources/model/PLaMo2-31B-kv-quant"
MODEL_PREFIX="plamo2-31b-fake-kv-quant"
DTYPE="bfloat16"
NUM_TRIALS=3

run() {
    local model_suffix=$1
    shift
    local cmd=(
        python run-hf.py
        --dtype "$DTYPE"
        --model "${MODEL_PREFIX}/${model_suffix}"
        --num-trials "$NUM_TRIALS"
        --path "$PATH_TO_MODEL"
        "$@"
    )
    echo ">>> ${cmd[*]}"
    "${cmd[@]}"
}

run "baseline"
run "fp8-static-1-0" --qbits 8 --qtype float --qstrategy static
run "int4-static-1-0" --qbits 4 --qtype int --qstrategy static
run "int4-token" --qbits 4 --qtype int --qstrategy token
run "int4-channel-on-the-fly-8" --qbits 4 --qtype int --qstrategy channel --qgroup 8