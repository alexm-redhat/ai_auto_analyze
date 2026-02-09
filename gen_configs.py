from dataclasses import dataclass


@dataclass
class ClaudeConfig:
    model: str
    allowed_tools: list[str]
    perm_mode: str
    cwd: str | None

def profile_system_prompt():
    return "You are a benchmarking and profiling expert of LLMs that run via vLLM, SGLang and TRT"
