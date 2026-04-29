import os
import time
import asyncio
import argparse

from common.utils import setup_logging
from common.claude_utils import claude_run

from auto_analyze.analyze_configs import (
    add_analyze_args,
    load_config,
    build_claude_config,
)
from auto_analyze.analyze_prompts import (
    SummaryPDFPrompt,
    SUMMARY_PDF_FILE,
    MEDIAN_BLOCK_FILE,
    PERF_COMPARE_BLOCK_FILE,
    PLAN_FILE,
)

OUTPUT_PDF_FILE = SUMMARY_PDF_FILE


def find_output_files(output_dir):
    files = os.listdir(output_dir)

    transformer_blocks = sorted(
        f for f in files if f.endswith(MEDIAN_BLOCK_FILE)
    )
    assert len(transformer_blocks) >= 1, (
        "No *{} files found in {}".format(MEDIAN_BLOCK_FILE, output_dir)
    )

    cmp_files = [f for f in files if f.endswith(PERF_COMPARE_BLOCK_FILE)]
    assert len(cmp_files) == 1, (
        "Expected 1 *{} file in {}, found {}".format(
            PERF_COMPARE_BLOCK_FILE, output_dir, len(cmp_files)
        )
    )

    plan_files = [f for f in files if f.endswith(PLAN_FILE)]
    assert len(plan_files) == 1, (
        "Expected 1 *{} file in {}, found {}".format(
            PLAN_FILE, output_dir, len(plan_files)
        )
    )

    return (
        transformer_blocks,
        os.path.join(output_dir, cmp_files[0]),
        os.path.join(output_dir, plan_files[0]),
    )


def gen_summary_pdf_step_prompts(config):
    output_dir = config["claude_output_dir"]
    framework_names = [fw["name"] for fw in config["frameworks"]]
    framework_source_codes = [fw["source_code"] for fw in config["frameworks"]]

    transformer_blocks, cmp_file, plan_file = find_output_files(output_dir)

    summary_pdf_prompt = SummaryPDFPrompt(
        model=config["model"],
        precision=config["precision"],
        gpu_type=config["gpu_type"],
        transformer_blocks=transformer_blocks,
        cmp_file=cmp_file,
        plan_file=plan_file,
        target_framework=config["target_framework"],
        framework_names=framework_names,
        framework_source_codes=framework_source_codes,
        output_file=os.path.join(output_dir, OUTPUT_PDF_FILE),
    )

    return [summary_pdf_prompt.prompt()]


if __name__ == "__main__":
    setup_logging("summary_pdf")

    parser = argparse.ArgumentParser(description="Generate summary PDF")
    add_analyze_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    claude_config = build_claude_config(config)

    start_time = time.time()

    print("=== Summary PDF ===")
    asyncio.run(
        claude_run(
            claude_config,
            gen_summary_pdf_step_prompts(config),
        )
    )

    duration_time = time.time() - start_time
    print("FINISHED: duration = {:.1f}s".format(duration_time))
