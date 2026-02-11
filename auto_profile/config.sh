AUTO_PROFILE_DIR="auto_profile"

verify_cwd() {
    local cwd
    cwd="$(pwd)"

    local missing=0

    if [[ ! -d "$cwd/${AUTO_PROFILE_DIR}" ]]; then
        if [[ ! -d "$cwd/$dir" ]]; then
            echo "Error: Required directory '$dir' not found in $cwd" >&2
            missing=1
        fi
    fi

    if [[ $missing -ne 0 ]]; then
        return 1
    fi

    return 0
}

# Dirs
if ! verify_cwd; then
  exit 1
fi
BASE_DIR="$(pwd)"

RESULTS_DIR="results"
DATASETS_DIR="datasets"
TRACES_DIR="traces"

# GPU Types
B200="b200"
H200="h200"

# Docker
DOCKER_PORT=30000
DOCKER_HF_HUB_CACHE="/app/hf_hub_cache"
DOCKER_BASE_DIR="/app/ai_auto_perf_analysis"

# Framework names
SGL="sgl"
VLLM="vllm"
TRT="trt"

# Docker result dirs
DOCKER_AUTO_PROFILE_DIR="${DOCKER_BASE_DIR}/${AUTO_PROFILE_DIR}"
DOCKER_RESULTS_DIR="${DOCKER_AUTO_PROFILE_DIR}/${RESULTS_DIR}"
VLLM_DOCKER_RESULTS_DIR="${DOCKER_AUTO_PROFILE_DIR}/${VLLM}"
SGL_DOCKER_RESULTS_DIR="${DOCKER_AUTO_PROFILE_DIR}/${SGL}"
TRT_DOCKER_RESULTS_DIR="${DOCKER_AUTO_PROFILE_DIR}/${TRT}"

# General test configs
NUM_WARMUPS=2
NUM_WAVES=4
NUM_TRACE_ITERS=50

