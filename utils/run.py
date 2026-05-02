"""Run-management helpers shared across tasks/* training scripts.

`init_run` materializes the run directory and dumps reproducibility
metadata (args, git SHA, code snapshot). `save_checkpoint` /
`load_checkpoint` standardize checkpoint I/O so every task uses the
same filename and history-retention semantics.

Folder convention
-----------------
``--log_dir`` is the *base* directory for an experiment family
(``logs/video/kinetics``); ``--run_name`` is the subdirectory of a
single run. The resolved run dir is ``{log_dir}/{run_name}``. When
``run_name`` is missing, ``init_run`` falls back to ``${SLURM_JOB_ID}``
on JZ, then to a local timestamp — so two ad-hoc runs never clobber
each other.

After ``init_run`` returns, ``args.log_dir`` is rewritten to the
resolved run dir; downstream code keeps reading ``args.log_dir`` as
its single source of truth.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from typing import Any, Optional

import torch

from utils.housekeeping import zip_python_code


def init_run(args, *, save_code: bool = True) -> str:
    """Create the run dir, dump args + git state, snapshot the repo."""
    run_name = getattr(args, "run_name", None) or _default_run_name()
    run_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    args.log_dir = run_dir

    args_dict = {k: _jsonable(v) for k, v in vars(args).items()}
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, "args.txt"), "w") as f:
        print(args, file=f)

    sha, dirty = _git_state()
    with open(os.path.join(run_dir, "git_sha.txt"), "w") as f:
        f.write(f"sha={sha}\ndirty={dirty}\n")

    if save_code:
        zip_python_code(os.path.join(run_dir, "repo_state.zip"))

    return run_dir


def save_checkpoint(
    log_dir: str,
    state: dict,
    step: int,
    *,
    keep_history: bool = False,
) -> None:
    """Always overwrite ``checkpoint.pt`` (latest). When ``keep_history``,
    also write ``checkpoint_{step:07d}.pt`` so intermediate snapshots
    accumulate alongside it.
    """
    torch.save(state, os.path.join(log_dir, "checkpoint.pt"))
    if keep_history:
        torch.save(state, os.path.join(log_dir, f"checkpoint_{step:07d}.pt"))


def load_checkpoint(log_dir: str, map_location: Any = None) -> Optional[dict]:
    """Load ``{log_dir}/checkpoint.pt`` if present, else ``None``."""
    path = os.path.join(log_dir, "checkpoint.pt")
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


def _default_run_name() -> str:
    slurm_id = os.environ.get("SLURM_JOB_ID")
    if slurm_id:
        return f"jz_{slurm_id}"
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        ).decode().strip())
        return sha, dirty
    except Exception:
        return "unknown", False


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)
