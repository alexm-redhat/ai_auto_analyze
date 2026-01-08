import sys
import asyncio


from analyze_configs import claude_config, analyze_configs
from analyze_prompts import gen_analyze_prompts, gen_perf_compare_prompt
from claude_utils import claude_run


LOG_FILE = "run_log.txt"


class Tee(object):
    """A file-like object that writes to multiple files simultaneously."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure immediate writing

    def flush(self):
        for f in self.files:
            f.flush()


def gen_prompts():
    prompts = []
    block_files = []
    for analyze_config in analyze_configs:
        cur_prompts, block_file = gen_analyze_prompts(analyze_config)
        prompts.extend(cur_prompts)
        block_files.append(block_file)

    prompts.append(gen_perf_compare_prompt(analyze_configs, block_files))

    return prompts


if __name__ == "__main__":
    # Redirect output to file as well
    log_file = open(LOG_FILE, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    asyncio.run(claude_run(claude_config, gen_prompts()))
