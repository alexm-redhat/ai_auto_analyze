import json
from dataclasses import dataclass
from common.claude_utils import ClaudeConfig


@dataclass
class AnalyzeConfig:
    model: str
    precision: str
    gpu_type: str
    framework_name: str
    framework_source_code: str
    test_dir: str
    gpu_ops_filter: str = ""


def add_analyze_args(parser):
    parser.add_argument(
        "--config", required=True, type=str, help="Path to JSON config file"
    )


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def build_analyze_configs(config):
    return [
        AnalyzeConfig(
            model=config["model"],
            precision=config["precision"],
            gpu_type=config["gpu_type"],
            framework_name=fw["name"],
            framework_source_code=fw["source_code"],
            test_dir=fw["test_dir"],
            gpu_ops_filter=fw.get("gpu_ops_filter", ""),
        )
        for fw in config["frameworks"]
    ]


def build_claude_config(config):
    return ClaudeConfig(
        model="claude-opus-4-6[1m]",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd=config["claude_output_dir"],
    )
