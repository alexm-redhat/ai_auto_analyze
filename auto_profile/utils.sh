source auto_profile/config.sh

_log() {
  local level="$1"
  shift
  local message="$*"
  local timestamp

  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "$timestamp [$level] $message"
}

log_info() {
  _log INFO "$@"
}

log_warn() {
  _log WARN "$@"
}

log_error() {
  _log ERROR "$@" >&2
}

log_debug() {
  _log DEBUG "$@"
}

write_run_metadata() {
    local dir="$1"
    local docker_image="$2"

    if [[ -z "$dir" || -z "$docker_image" ]]; then
        echo "Usage: write_run_metadata <dir> <docker_image>" >&2
        return 1
    fi

    local outfile="${dir}/run_metadata.txt"
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"

    # Explicitly use hostname command (due to docker)
    local hostname
    hostname="$(cat /proc/sys/kernel/hostname 2>/dev/null || echo unknown)"

    local os_info="Unknown"
    if [[ -f /etc/os-release ]]; then
        os_info="$(grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '"')"
    else
        os_info="$(uname -srv)"
    fi

    local gpu_info="No GPU detected"
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_info="$(nvidia-smi --query-gpu=index,name --format=csv,noheader)"
    fi

    echo "RUN METADATA"
    echo "============"

    {
        echo "Docker Image:      $docker_image"
        echo "Execution Time:    $timestamp"
        echo "Hostname:          $hostname"
        echo "Operating System:  $os_info"
        echo "GPU(s):"
        echo "$gpu_info"
    } | tee "$outfile"
}

create_dir_if_missing() {
  local dir="$1"

  if [[ -z "$dir" ]]; then
    log_error "No directory path provided"
    return 1
  fi

  log_info "Checking directory: $dir"

  if [[ -d "$dir" ]]; then
    log_info "  -- Directory already exists: $dir"
    return 0
  fi

  log_info "  -- Directory does not exist, create: $dir"

  mkdir "$dir" || {
    log_error "  -- Failed to create directory: $dir"
    return 1
  }
}

run_and_log() {
    local logfile="$1"
    shift

    set +e
    set -o pipefail

    local old_term old_int
    old_term="$(trap -p TERM)"
    old_int="$(trap -p INT)"

    trap '' TERM INT

    {
        printf "RUN-CMD:"
        printf " %q" "$@"
        printf "\n"

        bash -c 'exec "$@"' bash "$@"
    } 2>&1 | tee "$logfile"

    local status=${PIPESTATUS[0]}

    eval "$old_term"
    eval "$old_int"

    return "$status"
}

remove_docker_if_exists() {
  local name="$1"

  if [[ -z "$name" ]]; then
    log_error "No container name provided"
    return 1
  fi
  
  if docker container inspect $name >/dev/null 2>&1; then
	  log_warn "Container $name already exists. Stopping and removing."
    docker container stop $name
	  docker container rm $name
  fi

}

_run_docker() {
  local name="$1"
  local image="$2"
  local cmd="$3"
  local extra_flags="$4"

  if [[ -z "${name}" ]]; then
    log_error "No container name provided"
    return 1
  fi
  
  if [[ -z "${image}" ]]; then
    log_error "No container image provided"
    return 1
  fi

  if [[ -z "${cmd}" ]]; then
    log_error "No container cmd provided"
    return 1
  fi

  docker run \
    -it \
    --rm \
    --runtime nvidia \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    --shm-size 32g \
    --gpus=all \
    -v ${BASE_DIR}:${DOCKER_BASE_DIR} \
    -v ${HF_HUB_CACHE}:"${DOCKER_HF_HUB_CACHE}" \
    --env "HF_HUB_CACHE=${DOCKER_HF_HUB_CACHE}" \
    --env "HF_TOKEN=${HF_TOKEN}" \
    -p ${DOCKER_PORT}:${DOCKER_PORT} \
    --name ${name} \
    --entrypoint /bin/bash \
    ${extra_flags} \
    ${image} \
    -c "cd ${DOCKER_BASE_DIR}; time ${cmd}"

}

run_docker() {
  local framework="$1"
  local image="$2"
  local extra_flags="${3:-}"
  local run_config="${4:-}"

  if [[ -z "${framework}" ]]; then
    log_error "No framework name provided"
    return 1
  fi

  if [[ -z "${image}" ]]; then
    log_error "No container image provided"
    return 1
  fi

  local name="${framework}_auto_profile_${USER}"

  local cmd="${AUTO_PROFILE_DIR}/${framework}/${framework}_bench.sh"
  if [[ -n "$run_config" ]]; then
    cmd="${cmd} ${run_config}"
  fi

  remove_docker_if_exists $name

  _run_docker ${name} ${image} "${cmd}" "${extra_flags}"
}

clean_dir_contents() {
    local dir="${1:-}"
    shift || true
    local skip_names=("$@")

    # Require argument
    if [[ -z "$dir" ]]; then
        echo "Error: No directory argument provided." >&2
        return 1
    fi

    # Must exist and be directory
    if [[ ! -d "$dir" ]]; then
        echo "Error: '$dir' is not a valid directory." >&2
        return 1
    fi

    # Resolve absolute path
    local abs_dir
    abs_dir="$(realpath "$dir")"

    # Protect dangerous paths
    local home_abs
    home_abs="$(realpath "$HOME")"

    case "$abs_dir" in
        "/"|"."|".."|"$home_abs")
            echo "Error: Refusing to clean dangerous directory: $abs_dir" >&2
            return 1
            ;;
    esac

    # Enable matching of hidden files
    shopt -s nullglob dotglob

    local contents=("$abs_dir"/*)
    local to_delete=()
    local item base skip

    for item in "${contents[@]}"; do
        base="$(basename "$item")"
        skip=0

        for name in "${skip_names[@]}"; do
            if [[ "$base" == "$name" ]]; then
                skip=1
                break
            fi
        done

        if (( skip )); then
            log_info "Skipping: $item"
        else
            to_delete+=("$item")
        fi
    done

    if (( ${#to_delete[@]} == 0 )); then
        echo "Directory already empty or only contains skipped items: $abs_dir"
        return 0
    fi

    log_info "Cleaning contents of: $abs_dir"
    for item in "${to_delete[@]}"; do
        echo "  Removing: $item"
    done

    rm -rf -- "${to_delete[@]}"

    echo "Done."
}

create_clean_dir() {
  local dir="$1"

  create_dir_if_missing ${dir}
  clean_dir_contents ${dir}
}

make_test_name() {
  local framework="$1"
  local model="$2"
  local num_gpus="$3"
  local concurrency="$4"
  local input_len="$5"
  local output_len="$6"
  local mode="${7:-}"

  # Validate required parameters
  if [[ -z "$framework" || -z "$model" || -z "$num_gpus" || -z "$concurrency" || -z "$input_len" || -z "$output_len" ]]; then
    echo "Error: missing required parameter" >&2
    echo "Usage: make_test_name <framework> <model> <num_gpus> <concurrency> <input_len> <output_len> [mode]" >&2
    return 1
  fi

  local name
  name="${framework}-${model}-tp_${num_gpus}-isl_${input_len}-osl_${output_len}-b_${concurrency}"

  if [[ -n "$mode" ]]; then
    name+="-mode_${mode}"
  fi

  echo "$name"
}

make_test_dirname() {
  local output_dir="$1"
  local test_filename="$2"

  # Validate required parameters
  if [[ -z "$output_dir" || -z "$test_filename" ]]; then
    echo "Error: missing required parameter" >&2
    echo "Usage: make_test_dirname <output_dir> <test_filename> " >&2
    return 1
  fi

  echo "${output_dir}/test-${test_filename}"
}

make_result_filename() {
  local output_dir="$1"
  local test_filename="$2"

  # Validate required parameters
  if [[ -z "$output_dir" || -z "$test_filename" ]]; then
    echo "Error: missing required parameter" >&2
    echo "Usage: make_result_filename <output_dir> <test_filename> " >&2
    return 1
  fi

  echo "${output_dir}/bench-${test_filename}.json"
}

make_prepare_log_filename() {
  local output_dir="$1"
  local test_filename="$2"

  # Validate required parameters
  if [[ -z "$output_dir" || -z "$test_filename" ]]; then
    echo "Error: missing required parameter" >&2
    echo "Usage: make_prepare_log_filename <output_dir> <test_filename> " >&2
    return 1
  fi

  echo "${output_dir}/prepare-log-${test_filename}.txt"
}

make_run_log_filename() {
  local output_dir="$1"
  local test_filename="$2"

  # Validate required parameters
  if [[ -z "$output_dir" || -z "$test_filename" ]]; then
    echo "Error: missing required parameter" >&2
    echo "Usage: make_run_log_filename <output_dir> <test_filename> " >&2
    return 1
  fi

  echo "${output_dir}/run-log-${test_filename}.txt"
}

make_run_log_profile_filename() {
  local output_dir="$1"
  local test_filename="$2"

  # Validate required parameters
  if [[ -z "$output_dir" || -z "$test_filename" ]]; then
    echo "Error: missing required parameter" >&2
    echo "Usage: make_run_log_filename <output_dir> <test_filename> " >&2
    return 1
  fi

  echo "${output_dir}/run-log-profile-${test_filename}.txt"
}

make_trace_file_prefix() {
  local output_dir="$1"
  local test_filename="$2"

  # Validate required parameters
  if [[ -z "$output_dir" || -z "$test_filename" ]]; then
    echo "Error: missing required parameter" >&2
    echo "Usage: make_trace_file_prefix <output_dir> <test_filename> " >&2
    return 1
  fi

  echo "${output_dir}/trace-${test_filename}"
}

make_trace_dirname() {
  local output_dir="$1"
  local test_filename="$2"

  # Validate required parameters
  if [[ -z "$output_dir" || -z "$test_filename" ]]; then
    echo "Error: missing required parameter" >&2
    echo "Usage: make_trace_dirname <output_dir> <test_filename> " >&2
    return 1
  fi

  echo "${output_dir}/trace-${test_filename}"
}

calc_start_iter() {
  if [[ $# -ne 3 ]]; then
    echo "Usage: calc_start_iter <num_warmups> <num_waves> <output_len>" >&2
    return 1
  fi

  local num_warmups="$1"
  local num_waves="$2"
  local output_len="$3"

  # Validate integers
  if ! [[ "$num_warmups" =~ ^[0-9]+$ && \
          "$num_waves"   =~ ^[0-9]+$ && \
          "$output_len"  =~ ^[0-9]+$ ]]; then
    echo "Error: all arguments must be non-negative integers" >&2
    return 1
  fi

  echo $(( (num_warmups * output_len) + ((num_waves / 2) * output_len) + (output_len / 2) ))
}

calc_finish_iter() {
  if [[ $# -ne 2 ]]; then
    echo "Usage: calc_finish_iter <start_iter> <offset>" >&2
    return 1
  fi

  local start_iter="$1"
  local offset="$2"

  # Validate integers
  if ! [[ "$start_iter" =~ ^[0-9]+$ && "$offset" =~ ^[0-9]+$ ]]; then
    echo "Error: start_iter and offset must be non-negative integers" >&2
    return 1
  fi

  echo $(( start_iter + offset ))
}

make_results_dir_name() {
  echo "results"
}

setup_results_dirs() {
  local run_config="$1"

  RESULTS_DIR="$(make_results_dir_name "$run_config")"

  DOCKER_RESULTS_DIR="${DOCKER_AUTO_PROFILE_DIR}/${RESULTS_DIR}"
  VLLM_DOCKER_RESULTS_DIR="${DOCKER_RESULTS_DIR}/${VLLM}"
  SGL_DOCKER_RESULTS_DIR="${DOCKER_RESULTS_DIR}/${SGL}"
  TRT_DOCKER_RESULTS_DIR="${DOCKER_RESULTS_DIR}/${TRT}"
}

load_run_config() {
  local run_config="$1"

  if [[ -z "$run_config" ]]; then
    log_error "Usage: load_run_config <run_config.json>"
    return 1
  fi

  if [[ ! -f "$INFRA_CONFIG" ]]; then
    log_error "Infra config file not found: $INFRA_CONFIG"
    return 1
  fi

  if [[ ! -f "$run_config" ]]; then
    log_error "Run config file not found: $run_config"
    return 1
  fi

  eval "$(python3 "${AUTO_PROFILE_DIR}/parse_run_config.py" "$INFRA_CONFIG" "$run_config")"

  setup_results_dirs "$run_config"

  log_info "Loaded config: infra=$INFRA_CONFIG run=$run_config"
  log_info "  RESULTS_DIR=${RESULTS_DIR}"
  log_info "  VLLM_DOCKER_IMAGE=${VLLM_DOCKER_IMAGE}"
  log_info "  SGL_DOCKER_IMAGE=${SGL_DOCKER_IMAGE}"
  log_info "  TRT_DOCKER_IMAGE=${TRT_DOCKER_IMAGE}"
  log_info "  PROFILES=( ${PROFILES[*]} )"
  for _p in "${PROFILES[@]}"; do
    log_info "    model=${_p}"
    log_info "      gpu_ids=${PROFILE_GPU_IDS[$_p]}"
    log_info "      input_len=${PROFILE_INPUT_LENS[$_p]} output_len=${PROFILE_OUTPUT_LENS[$_p]}"
    log_info "      vllm_mode=${PROFILE_VLLM_MODES[$_p]} sgl_mode=${PROFILE_SGL_MODES[$_p]} trt_mode=${PROFILE_TRT_MODES[$_p]}"
  done
  log_info "  PROFILE_CONCURRENCIES=${PROFILE_CONCURRENCIES}"
  log_info "  RUN_FRAMEWORKS=${RUN_FRAMEWORKS}"
  log_info "  ENABLE_TRACES=${ENABLE_TRACES}"
}

is_framework_enabled() {
  local fw="$1"
  [[ " ${RUN_FRAMEWORKS} " == *" ${fw} "* ]]
}

is_trace_enabled() {
  local fw="$1"
  [[ " ${ENABLE_TRACES} " == *" ${fw} "* ]]
}

find_nsys_dir() {
  local nsys_path nsys_install_dir

  # 1) Respect NSYS_HOME if user provided it
  if [[ -n "${NSYS_HOME:-}" && -x "${NSYS_HOME}/bin/nsys" ]]; then
    echo "$(readlink -f "${NSYS_HOME}")"
    return 0
  fi

  # 2) Prefer real installs under /opt/nvidia/nsight-systems
  # Pick the newest version directory if multiple exist
  nsys_path="$(
    ls -1d /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"

  if [[ -n "${nsys_path}" && -x "${nsys_path}" ]]; then
    nsys_install_dir="$(dirname "$(dirname "${nsys_path}")")"
    echo "$(readlink -f "${nsys_install_dir}")"
    return 0
  fi

  # 3) Fallback: use PATH, but reject CUDA Toolkit wrapper locations
  nsys_path="$(command -v nsys 2>/dev/null || true)"
  if [[ -z "${nsys_path}" ]]; then
    echo "nsys not found in PATH and no /opt/nvidia/nsight-systems install found" >&2
    return 1
  fi

  nsys_path="$(readlink -f "${nsys_path}")"

  # Reject the common CUDA Toolkit wrapper path(s)
  if [[ "${nsys_path}" == /usr/local/cuda*/bin/nsys ]]; then
    echo "Found nsys wrapper at ${nsys_path}, but no /opt/nvidia/nsight-systems install found" >&2
    return 1
  fi

  nsys_install_dir="$(dirname "$(dirname "${nsys_path}")")"
  echo "${nsys_install_dir}"
}
