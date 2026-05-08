import os
import json
from dataclasses import dataclass, field, asdict

from auto_analyze.single_trace_config import (
    MEDIAN_BLOCK_FILE,
    HIGH_LEVEL_OPS_FILE,
)


PERF_COMPARE_FILE = "perf_compare_blocks.txt"
PERF_DIFF_ANALYSIS_FILE = "perf_diff_analysis.txt"
IMPROVEMENT_PLAN_FILE = "improvement_plan.txt"

CROSS_TRACE_OUTPUT_FILES = {
    "perf_compare": PERF_COMPARE_FILE,
    "perf_diff_analysis": PERF_DIFF_ANALYSIS_FILE,
    "improvement_plan": IMPROVEMENT_PLAN_FILE,
}

ANALYSIS_TYPE_CROSS_FRAMEWORK = "cross-framework"
ANALYSIS_TYPE_REGRESSION = "regression"
VALID_ANALYSIS_TYPES = (ANALYSIS_TYPE_CROSS_FRAMEWORK, ANALYSIS_TYPE_REGRESSION)


@dataclass
class TraceResult:
    trace_id: str
    framework_name: str
    framework_source_code: str
    result_dir: str = ""
    median_block_file: str = ""
    high_level_ops_file: str = ""

    def get_median_block_file(self):
        if self.median_block_file:
            return self.median_block_file
        return os.path.join(self.result_dir, MEDIAN_BLOCK_FILE)

    def get_high_level_ops_file(self):
        if self.high_level_ops_file:
            return self.high_level_ops_file
        return os.path.join(self.result_dir, HIGH_LEVEL_OPS_FILE)


@dataclass
class CrossTraceConfig:
    trace_results: list[TraceResult]
    analysis_type: str
    target_trace_id: str
    model: str
    gpu_type: str
    output_dir: str = "./cross_trace_output"

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        trace_results = [
            TraceResult(**tr) for tr in data.pop("trace_results", [])
        ]
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(trace_results=trace_results, **filtered)

    def to_json(self, path):
        d = asdict(self)
        with open(path, "w") as f:
            json.dump(d, f, indent=4)

    def get_target_result(self):
        for tr in self.trace_results:
            if tr.trace_id == self.target_trace_id:
                return tr
        raise ValueError(
            f"target_trace_id '{self.target_trace_id}' not found in trace_results. "
            f"Available: {[tr.trace_id for tr in self.trace_results]}"
        )

    def validate(self):
        errors = []
        if not self.model:
            errors.append("model is required")
        if not self.gpu_type:
            errors.append("gpu_type is required")
        if self.analysis_type not in VALID_ANALYSIS_TYPES:
            errors.append(
                f"analysis_type must be one of {VALID_ANALYSIS_TYPES}, "
                f"got '{self.analysis_type}'"
            )
        if len(self.trace_results) < 2:
            errors.append("at least 2 trace_results are required")
        if not self.target_trace_id:
            errors.append("target_trace_id is required")

        trace_ids = [tr.trace_id for tr in self.trace_results]
        if self.target_trace_id and self.target_trace_id not in trace_ids:
            errors.append(
                f"target_trace_id '{self.target_trace_id}' not found. "
                f"Available: {trace_ids}"
            )

        for i, tr in enumerate(self.trace_results):
            if not tr.trace_id:
                errors.append(f"trace_results[{i}].trace_id is required")
            if not tr.framework_name:
                errors.append(f"trace_results[{i}].framework_name is required")
            if not tr.framework_source_code:
                errors.append(
                    f"trace_results[{i}].framework_source_code is required"
                )

            median_file = tr.get_median_block_file()
            if not os.path.exists(median_file):
                errors.append(
                    f"trace_results[{i}] median block file not found: {median_file}"
                )
            high_level_file = tr.get_high_level_ops_file()
            if not os.path.exists(high_level_file):
                errors.append(
                    f"trace_results[{i}] high-level ops file not found: {high_level_file}"
                )

        return errors
