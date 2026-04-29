#!/bin/bash

INFRA_CONFIG="${1:?Usage: $0 <infra_config.json> <run_config.json>}"
RUN_CONFIG="${2:?Usage: $0 <infra_config.json> <run_config.json>}"

source auto_profile/utils.sh
load_run_config "$INFRA_CONFIG" "$RUN_CONFIG"

nsys_flags=""
if is_trace_enabled vllm; then
    nsys_dir=$(find_nsys_dir)
    nsys_flags="-v ${nsys_dir}:/nsys:ro -e PATH=/nsys/bin:${PATH}"
fi

run_docker ${VLLM} ${VLLM_DOCKER_IMAGE} "${nsys_flags}" "${INFRA_CONFIG}" "${RUN_CONFIG}"
