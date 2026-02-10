
clear_directory_contents() {
    local BASE_DIR="$1"

    if [[ -z "${BASE_DIR:-}" ]]; then
        echo "Error: No directory argument provided" >&2
        return 1
    fi

    echo "Preparing to clear contents of: ${BASE_DIR}"

    # Quick dangerous input checks
    case "${BASE_DIR}" in
        "/"|"."|".."|"~"|"$HOME")
            echo "Error: Refusing to delete dangerous directory: ${BASE_DIR}" >&2
            return 1
            ;;
    esac

    if [[ ! -d "${BASE_DIR}" ]]; then
        echo "Error: Directory does not exist: ${BASE_DIR}" >&2
        return 1
    fi

    # Resolve absolute paths
    local ABS_PATH HOME_ABS
    ABS_PATH="$(realpath "${BASE_DIR}")"
    HOME_ABS="$(realpath "${HOME}")"

    # Block dangerous resolved paths
    if [[ "${ABS_PATH}" == "/" || "${ABS_PATH}" == "${HOME_ABS}" ]]; then
        echo "Error: Refusing to delete dangerous directory: ${ABS_PATH}" >&2
        return 1
    fi

    shopt -s nullglob dotglob
    local files=("${ABS_PATH}"/*)

    if (( ${#files[@]} == 0 )); then
        echo "Directory already empty."
        return 0
    fi

    echo "The following items will be deleted:"
    for f in "${files[@]}"; do
        echo "  - ${f}"
    done

    echo "Clearing contents of ${ABS_PATH}"
    rm -rf -- "${files[@]}"

    echo "Done."
}