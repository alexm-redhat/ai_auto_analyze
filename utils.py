from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


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


def clear_vllm_source_tree(
    source_dir: str | Path,
    *,
    remove_ignored: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Reset a vLLM source tree to a clean git state.

    This removes:
    - tracked modifications
    - staged changes
    - untracked files
    - untracked directories

    Parameters
    ----------
    source_dir:
        Path to the vLLM git repository or any directory inside it.
    remove_ignored:
        If True, also remove ignored files/directories (equivalent to `git clean -fdx`).
    dry_run:
        If True, only print what would be removed.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    NotADirectoryError
        If the path is not a directory.
    RuntimeError
        If git is missing, the path is not a git repo, or a git command fails.
    """
    source_dir = Path(source_dir).expanduser().resolve()

    if shutil.which("git") is None:
        raise RuntimeError("git is not installed or not in PATH")

    if not source_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {source_dir}")

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {source_dir}")

    repo_check = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=source_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if repo_check.returncode != 0 or repo_check.stdout.strip() != "true":
        raise RuntimeError(f"Not a git repository: {source_dir}")

    root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=source_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if root_result.returncode != 0:
        raise RuntimeError(
            f"Failed to determine repository root:\n{root_result.stderr.strip()}"
        )

    repo_root = Path(root_result.stdout.strip()).resolve()

    if dry_run:
        clean_cmd = ["git", "clean", "-fdn"]
        if remove_ignored:
            clean_cmd = ["git", "clean", "-fdxn"]

        print(f"[dry-run] Repo root: {repo_root}")
        print("[dry-run] Would run: git reset --hard")
        print(f"[dry-run] Would run: {' '.join(clean_cmd)}")

        result = subprocess.run(
            clean_cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.returncode != 0:
            raise RuntimeError(f"Dry-run clean failed:\n{result.stderr.strip()}")
        return

    reset_result = subprocess.run(
        ["git", "reset", "--hard"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if reset_result.returncode != 0:
        raise RuntimeError(f"`git reset --hard` failed:\n{reset_result.stderr.strip()}")

    clean_cmd = ["git", "clean", "-fdx"] if remove_ignored else ["git", "clean", "-fd"]
    clean_result = subprocess.run(
        clean_cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if clean_result.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(clean_cmd)}` failed:\n{clean_result.stderr.strip()}"
        )
