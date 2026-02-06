#!/bin/bash

echo "RUN VLLM DOCKER"
echo "==============="
./vllm/vllm_run_docker.sh

echo "RUN SGL DOCKER"
echo "==============="
./sgl/sgl_run_docker.sh

echo "RUN TRT DOCKER"
echo "==============="
./trt/trt_run_docker.sh
