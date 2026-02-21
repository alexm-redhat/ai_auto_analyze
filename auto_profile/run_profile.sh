#!/bin/bash

echo "RUN SGL DOCKER"
echo "==============="
time ./auto_profile/sgl/sgl_run_docker.sh

echo "RUN VLLM DOCKER"
echo "==============="
time ./auto_profile/vllm/vllm_run_docker.sh

echo "RUN TRT DOCKER"
echo "==============="
time ./auto_profile/trt/trt_run_docker.sh


