import os


from analyze_configs import claude_config, VLLM_TRACE_FILE, SGLANG_TRACE_FILE


def clear_claude_cwd():
    cwd = claude_config.cwd
    skip_files = [VLLM_TRACE_FILE, SGLANG_TRACE_FILE]

    print("Clearing Claude CWD = {}".format(cwd))
    for filename in os.listdir(cwd):
        if filename in skip_files:
            print("    Skip filename = {}".format(filename))
            continue

        path = os.path.join(cwd, filename)

        if os.path.isdir(path):
            print("    Skip directory = {}".format(filename))
            continue

        print("    Delete filename = {}".format(filename))
        os.unlink(path)


if __name__ == "__main__":
    clear_claude_cwd()
