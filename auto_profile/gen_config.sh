verify_profile_dir() {
  local cwd base required_file

  cwd="$PWD"
  base="$(basename -- "$cwd")"
  required_file="$cwd/profile_config.sh"

  if [[ "$base" != "profile" ]]; then
    echo "Error: must be run from a directory named 'profile' (found: '$base')" >&2
    return 1
  fi

  if [[ ! -f "$required_file" ]]; then
    echo "Error: missing required file: profile_config.sh" >&2
    return 1
  fi

  return 0
}

# Output dirs
PROFILE_DIR="$PWD"
if ! verify_profile_dir; then
  exit 1
fi

RESULTS_DIR="results"
DATASETS_DIR="datasets"
TRACES_DIR="traces"

# GPU Types
B200="b200"

# Docker
DOCKER_PORT=30000
DOCKER_HF_HUB_CACHE="/app/hf_hub_cache"
DOCKER_PROFILE_DIR="/app/profile"

# Framework names
SGL="sgl"
VLLM="vllm"
TRT="trt"

# Docker result dirs
VLLM_DOCKER_RESULTS_DIR="${DOCKER_PROFILE_DIR}/${RESULTS_DIR}/${VLLM}"
SGL_DOCKER_RESULTS_DIR="${DOCKER_PROFILE_DIR}/${RESULTS_DIR}/${SGL}"
TRT_DOCKER_RESULTS_DIR="${DOCKER_PROFILE_DIR}/${RESULTS_DIR}/${TRT}"

# General test configs
NUM_WARMUPS=2
NUM_WAVES=4
NUM_TRACE_ITERS=50

