#!/bin/bash

set -e

run_and_eval() {
    YAML=$1
    OUTFILE=$2

    # Start server in background, redirect output to a log file
    LOGFILE="server.log"
    rm -f "$LOGFILE"
    llamafactory-cli api "$YAML" > "$LOGFILE" 2>&1 &
    SERVER_PID=$!

    # Wait for server to be ready
    echo "Waiting for server to start..."
    while ! grep -q "Uvicorn running on http://0.0.0.0:8000" "$LOGFILE"; do
        sleep 1
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server process exited unexpectedly."
            exit 1
        fi
    done
    echo "Server started."

    # Run eval script
    OUTFILE="$OUTFILE" python scripts/eval_prefecture_qwen.py

    # Kill server
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null || true
    echo "Server stopped."
}

# run_and_eval examples/inference/qwen2_5vl.yaml data/eval_results_qwen3b_jp_sampled.json
# llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft_jp.yaml
run_and_eval examples/inference/qwen2_5vl_jp.yaml data/eval_results_lora_jp_sampled_no_tokyo.json
# run_and_eval examples/inference/qwen2_5vl_7b.yaml data/eval_results_qwen7b_jp_sampled.json
# llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft_7b.yaml
# run_and_eval examples/inference/qwen2_5vl_jp_7b.yaml data/eval_results_lora_7b_jp_sampled_p02.json 