import os
import gzip
import json
import shutil
import subprocess
from dataclasses import dataclass, asdict, fields


HIGH_LEVEL_OPS_FILE = "transformer_block_high_level_ops.txt"
GPU_OPS_TXT_FILE = "gpu_ops.txt"
GPU_OPS_TO_BLOCKS_FILE = "gpu_ops_to_blocks.txt"
MEDIAN_BLOCK_FILE = "median_block.txt"
PERF_ANALYSIS_SINGLE_FILE = "perf_analysis_single_trace.txt"

SINGLE_TRACE_OUTPUT_FILES = {
    "high_level_ops": HIGH_LEVEL_OPS_FILE,
    "gpu_ops_txt": GPU_OPS_TXT_FILE,
    "gpu_ops_to_blocks": GPU_OPS_TO_BLOCKS_FILE,
    "median_block": MEDIAN_BLOCK_FILE,
    "perf_analysis": PERF_ANALYSIS_SINGLE_FILE,
}

MAX_GPU_OPS = 2000

TRACE_FILE_TYPE_NSYS = "NSYS"
TRACE_FILE_TYPE_PYTORCH = "PYTORCH"
VALID_TRACE_FILE_TYPES = (TRACE_FILE_TYPE_NSYS, TRACE_FILE_TYPE_PYTORCH)

TRACE_GPU_FOCUS_ALL = "ALL"

VALID_TRACE_EXTENSIONS = (".nsys-rep", ".sqlite", ".json", ".json.gz")



@dataclass
class SingleTraceParams:
    """All parameters from the single-trace config JSON that prompts use."""

    framework_name: str
    model: str
    gpu_type: str
    batch_size_range: str
    prefill_size_range: str
    output_size_range: str
    trace_file: str
    framework_source_code: str
    trace_gpu_focus: str
    run_command: str
    commit_id: str = "HEAD"
    run_log: str = ""
    high_level_focus: str = ""
    perf_analysis_focus: str | None = None
    skip_perf_analysis: bool = False
    max_gpu_ops: int = MAX_GPU_OPS

    @property
    def trace_file_type(self):
        if self.trace_file.endswith(".sqlite") or self.trace_file.endswith(".nsys-rep"):
            return TRACE_FILE_TYPE_NSYS
        if self.trace_file.endswith(".json") or self.trace_file.endswith(".json.gz"):
            return TRACE_FILE_TYPE_PYTORCH
        return TRACE_FILE_TYPE_NSYS

    def context_header(self):
        lines = [
            f"<model> = {self.model}",
            f"<model_hf_url> = https://huggingface.co/{self.model}",
            f"<gpu_type> = {self.gpu_type}",
            f"<framework_name> = {self.framework_name}",
            f"<framework_source_code> = {self.framework_source_code}",
            f"<trace_file> = {self.trace_file}",
            f"<trace_file_type> = {self.trace_file_type}",
            f"<trace_gpu_focus> = {self.trace_gpu_focus.strip()}",
            f"<batch_size_range> = {self.batch_size_range}",
            f"<prefill_size_range> = {self.prefill_size_range}",
            f"<output_size_range> = {self.output_size_range}",
            f"<max_gpu_ops> = {self.max_gpu_ops}",
            f"<run_command> = {self.run_command}",
        ]
        if self.run_log:
            lines.append(f"<run_log> = {self.run_log}")
        if self.high_level_focus:
            lines.append(f"<high_level_focus> = {self.high_level_focus}")
        if hasattr(self, "output_dir"):
            lines.append(f"<output_dir> = {os.path.abspath(self.output_dir)}")
        lines.append(
            "IMPORTANT: Write all output files to the absolute paths"
            " specified below. All paths are under <output_dir>."
            " Do not create files anywhere else."
        )
        return "\n".join(lines)

    def gpu_focus_instruction(self):
        if self.trace_gpu_focus == TRACE_GPU_FOCUS_ALL:
            return (
                "For below, analyze GPU operations across ALL GPUs"
                " in the trace:"
            )
        return (
            f"For below, focus only on GPU {self.trace_gpu_focus}"
            f" and ignore other GPUs:"
        )



@dataclass
class SingleTraceConfig(SingleTraceParams):
    """SingleTraceParams plus operational fields and methods."""

    output_dir: str = "./single_trace_output"

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    def save_result_metadata(self):
        metadata = {
            "framework_name": self.framework_name,
            "model": self.model,
            "gpu_type": self.gpu_type,
            "framework_source_code": self.framework_source_code,
            "commit_id": self.commit_id,
            "output_dir": os.path.abspath(self.output_dir),
            "output_files": SINGLE_TRACE_OUTPUT_FILES,
        }
        path = os.path.join(self.output_dir, "result_metadata.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)

    def prepare_source_code(self):
        src = self.framework_source_code
        commit = self.commit_id
        fw = self.framework_name

        print(f"  [source] Preparing source code: {src}")
        print(f"  [source] Target commit: {commit}")

        # Check it's a git repo
        git_dir = os.path.join(src, ".git")
        if not os.path.exists(git_dir):
            raise RuntimeError(
                f"framework_source_code is not a git repository: {src}\n"
                f"  Expected .git directory at: {git_dir}"
            )
        print(f"  [source] Git repository confirmed: {src}")

        def _git(*args):
            result = subprocess.run(
                ["git", "-C", src] + list(args),
                capture_output=True, text=True,
            )
            return result

        # Check for uncommitted changes that would block branch switching
        status = _git("status", "--porcelain")
        if status.returncode != 0:
            raise RuntimeError(
                f"git status failed in {src}:\n{status.stderr}"
            )
        dirty_lines = [
            line for line in status.stdout.strip().splitlines()
            if line and not line.startswith("??")
        ]
        if dirty_lines:
            raise RuntimeError(
                f"framework_source_code has uncommitted changes that would block branch switching:\n"
                f"  Directory: {src}\n"
                f"  Changes:\n" +
                "".join(f"    {line}\n" for line in dirty_lines) +
                f"\n  Please commit or stash your changes before running the analysis.\n"
                f"  You can run: git -C {src} stash"
            )
        print(f"  [source] Working tree is clean (no uncommitted changes)")

        # Resolve commit to full hash
        resolve = _git("rev-parse", commit)
        if resolve.returncode != 0:
            raise RuntimeError(
                f"commit_id '{commit}' could not be resolved in {src}.\n"
                f"  You may need to run: git -C {src} fetch --all"
            )
        full_hash = resolve.stdout.strip()
        self.commit_id = full_hash
        print(f"  [source] Resolved commit: {full_hash[:12]}")

        if commit.upper() == "HEAD":
            # HEAD means use the current commit — no branch switching needed
            current_branch = _git("branch", "--show-current").stdout.strip()
            print(f"  [source] Using HEAD on current branch: {current_branch} ({full_hash[:12]})")
        else:
            # Specific commit requested — set up a dedicated branch
            short_hash = full_hash[:6]
            branch_name = f"auto_analyze_{fw}_commit_{short_hash}"
            print(f"  [source] Target branch: {branch_name}")

            branch_check = _git("rev-parse", "--verify", f"refs/heads/{branch_name}")
            if branch_check.returncode == 0:
                print(f"  [source] Branch '{branch_name}' already exists, switching to it")
                checkout = _git("checkout", branch_name)
                if checkout.returncode != 0:
                    raise RuntimeError(
                        f"Failed to checkout branch '{branch_name}':\n{checkout.stderr}"
                    )
                head = _git("rev-parse", "HEAD").stdout.strip()
                if not head.startswith(full_hash[:12]):
                    raise RuntimeError(
                        f"Branch '{branch_name}' exists but its HEAD ({head[:12]}) "
                        f"does not match commit_id ({full_hash[:12]}).\n"
                        f"  Delete the branch and retry:\n"
                        f"    git -C {src} branch -D {branch_name}"
                    )
                print(f"  [source] Verified: branch HEAD matches commit_id ({head[:12]})")
            else:
                print(f"  [source] Creating branch '{branch_name}' at commit {full_hash[:12]}")
                create = _git("checkout", "-b", branch_name, full_hash)
                if create.returncode != 0:
                    raise RuntimeError(
                        f"Failed to create branch '{branch_name}' at {full_hash[:12]}:\n"
                        f"{create.stderr}"
                    )
            print(f"  [source] Branch created and checked out")

        # Final confirmation
        current_branch = _git("branch", "--show-current").stdout.strip()
        current_head = _git("rev-parse", "--short", "HEAD").stdout.strip()
        print(f"  [source] Ready: branch={current_branch}, HEAD={current_head}")

    def prepare_trace_file(self):
        if self.trace_file.endswith(".nsys-rep"):
            src = self.trace_file
            dst = os.path.join(
                self.output_dir,
                os.path.basename(src).rsplit(".", 1)[0] + ".sqlite",
            )
            print(f"  Converting .nsys-rep to .sqlite ...")
            print(f"    Source:  {src}")
            print(f"    Output:  {dst}")
            if shutil.which("nsys") is None:
                raise RuntimeError(
                    "nsys not found in PATH. Install NVIDIA Nsight Systems "
                    "or convert manually:\n"
                    f'  nsys export --type=sqlite --force-overwrite true --output="{dst}" "{src}"'
                )
            subprocess.run(
                ["nsys", "export", "--type=sqlite", "--force-overwrite", "true", f"--output={dst}", src],
                check=True,
            )
            self.trace_file = dst
            print(f"    Done.  trace_file updated to: {dst}")

        elif self.trace_file.endswith(".json.gz"):
            src = self.trace_file
            dst = os.path.join(
                self.output_dir,
                os.path.basename(src).rsplit(".gz", 1)[0],
            )
            print(f"  Decompressing .json.gz ...")
            print(f"    Source:  {src}")
            print(f"    Output:  {dst}")
            with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            self.trace_file = dst
            print(f"    Done.  trace_file updated to: {dst}")

    def validate(self):
        errors = []
        if not self.framework_name:
            errors.append("framework_name is required")
        if not self.model:
            errors.append("model is required")
        if not self.gpu_type:
            errors.append("gpu_type is required")
        if not self.batch_size_range:
            errors.append("batch_size_range is required")
        if not self.prefill_size_range:
            errors.append("prefill_size_range is required")
        if not self.output_size_range:
            errors.append("output_size_range is required")
        if not self.run_command:
            errors.append("run_command is required")

        if not self.trace_file:
            errors.append("trace_file is required")
        elif not any(
            self.trace_file.endswith(ext) for ext in VALID_TRACE_EXTENSIONS
        ):
            errors.append(
                f"trace_file must end with one of {VALID_TRACE_EXTENSIONS}, "
                f'got "{self.trace_file}".'
            )
        elif not os.path.exists(self.trace_file):
            errors.append(f"trace_file not found: {self.trace_file}")

        if not self.framework_source_code:
            errors.append("framework_source_code is required")
        elif not os.path.isdir(self.framework_source_code):
            errors.append(
                "framework_source_code not a directory:"
                f" {self.framework_source_code}"
            )
        if self.run_log and not os.path.exists(self.run_log):
            errors.append(f"run_log not found: {self.run_log}")

        focus = self.trace_gpu_focus.strip()
        if focus != TRACE_GPU_FOCUS_ALL:
            if not focus.isdigit():
                errors.append(
                    f'trace_gpu_focus must be "ALL" or a non-negative '
                    f'integer GPU ID, got "{self.trace_gpu_focus}"'
                )

        return errors
