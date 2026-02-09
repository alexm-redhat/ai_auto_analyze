import os


from analyze_configs import claude_config


def clear_claude_cwd():
    cwd = claude_config.cwd

    print("Clearing Claude CWD = {}".format(cwd))
    for filename in os.listdir(cwd):
        path = os.path.join(cwd, filename)

        if os.path.isdir(path):
            print("    Skip directory = {}".format(filename))
            continue

        print("    Delete filename = {}".format(filename))
        os.unlink(path)


if __name__ == "__main__":
    clear_claude_cwd()
