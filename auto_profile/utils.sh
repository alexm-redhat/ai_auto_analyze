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

detect_container_engine() {
  # If docker exists as a real executable, use it
  if command -v docker >/dev/null 2>&1; then
    # Check if docker resolves to podman binary
    local resolved
    resolved="$(command -v docker)"

    if [[ "$resolved" == *podman* ]]; then
      echo "podman"
      return 0
    fi

    # If it's a real docker binary
    echo "docker"
    return 0
  fi

  # If docker not found but podman exists
  if command -v podman >/dev/null 2>&1; then
    echo "podman"
    return 0
  fi

  echo "Error: Neither docker nor podman found in PATH" >&2
  return 1
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
  
  engine="$(detect_container_engine)" || exit 1

  # --ipc=host \ TODO: Remove

  #--userns=keep-id \
  #--security-opt label=disable \
  # --device nvidia.com/gpu=all \
  # -v /home/alexm-redhat/vllm_workspace:/vllm-workspace \
  # --env "HF_HOME=${DOCKER_HF_HUB_CACHE}" \
  docker run \
    -it \
    --rm \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    --shm-size 32g \
    --gpus=all \
    -v ${BASE_DIR}:${DOCKER_BASE_DIR} \
    -v ${HF_HUB_CACHE}:"${DOCKER_HF_HUB_CACHE}" \
    --env "HF_HUB_CACHE=${DOCKER_HF_HUB_CACHE}" \
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
  local extra_flags="$3"

  if [[ -z "${framework}" ]]; then
    log_error "No framework name provided"
    return 1
  fi
  
  if [[ -z "${image}" ]]; then
    log_error "No container image provided"
    return 1
  fi

  local name="${framework}_auto_profile_${USER}"
  
  # source ${framework}/${framework}_config.sh
  
  remove_docker_if_exists $name

  _run_docker ${name} ${image} "${AUTO_PROFILE_DIR}/${framework}/${framework}_bench.sh" "${extra_flags}"
}

clean_dir_contents() {
    local dir="${1:-}"

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

    if (( ${#contents[@]} == 0 )); then
        echo "Directory already empty: $abs_dir"
        return 0
    fi

    log_info "Cleaning contents of: $abs_dir"
    for item in "${contents[@]}"; do
        echo "  Removing: $item"
    done

    rm -rf -- "${contents[@]}"

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

is_vllm_profile_enabled() {
  [[ ${VLLM_ENABLE_PROFILE:-0} == 1 ]]
}

is_sgl_profile_enabled() {
  [[ ${SGL_ENABLE_PROFILE:-0} == 1 ]]
}

is_trt_profile_enabled() {
  [[ ${TRT_ENABLE_PROFILE:-0} == 1 ]]
}

find_nsys_dir() {
  nsys_path=$(readlink -f "$(which nsys)") || {
    echo "nsys not found in PATH" >&2
    exit 1
  }

  nsys_bin_dir=$(dirname "$nsys_path")
  nsys_install_dir=$(dirname "$nsys_bin_dir")

  echo ${nsys_install_dir}
}