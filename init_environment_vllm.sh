#!/usr/bin/env bash

SELECTOR="${1:-vllm_0_12}"

echo "initializing vLLM environment: $SELECTOR"
uname -a

case "$SELECTOR" in
    vllm_0_12)
        ml vLLM/0.12.0-foss-2025a-CUDA-12.8.0
        source ~/virtualenvs/vllm_0_12/bin/activate
        ;;
    vllm_0_8_5)
        ml vLLM/0.8.5.post1-foss-2024a-CUDA-12.6.0
        source ~/virtualenvs/vllm_0_8_5/bin/activate
        ;;
    *)
        echo "Unknown vLLM selector: $SELECTOR"
        echo "Supported: vllm_0_12, vllm_0_8_5"
        exit 1
        ;;
esac

echo "done"