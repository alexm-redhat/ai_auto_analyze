#!/bin/bash

echo "RUN VLLM DOCKER"
echo "==============="
time ./vllm/vllm_run_docker.sh

echo "RUN SGL DOCKER"
echo "==============="
time ./sgl/sgl_run_docker.sh

echo "RUN TRT DOCKER"
echo "==============="
time ./trt/trt_run_docker.sh
