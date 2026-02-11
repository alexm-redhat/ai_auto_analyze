#!/bin/bash

source auto_profile/utils.sh
source auto_profile/profile_config.sh

nsys_flags=""
if is_vllm_profile_enabled; then
    nsys_dir=$(find_nsys_dir)
    nsys_flags="-v ${nsys_dir}:/nsys:ro -e PATH=/nsys/bin:${PATH}"
    
fi

run_docker ${VLLM} ${VLLM_DOCKER_IMAGE} "${nsys_flags}"
