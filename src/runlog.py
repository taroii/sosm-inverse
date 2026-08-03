"""
Run directories, provenance, and metrics. Pure Python -- no Firedrake.

Implements the pipeline in notes/server.md: every run gets its own directory
holding the exact config, full provenance, logs, and a metrics CSV. Bulk output
(VTK, checkpoints) goes in `output/` and never enters git. Results are promoted
into a tracked `results/` only when they become paper figures.

Usage:

    from runlog import Run

    with Run("manufactured-2d-k4", config, seed=0) as run:
        ...
        run.record(N_mesh=8, mu_1_err=1.2e-4)

The provenance requirement is not hygiene, it is README.md section IV: "fixed
seeds and versioned configuration files for every reported run".
"""

import csv
import json
import hashlib
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["Run", "repo_root", "provenance"]


def repo_root():
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return here.parent


def _git(*args, cwd=None):
    try:
        out = subprocess.run(["git", *args], cwd=cwd or repo_root(),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _package_version(name):
    try:
        module = __import__(name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def provenance(extra=None):
    """Everything needed to reproduce or defend a run."""
    root = repo_root()

    info = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "firedrake": _package_version("firedrake"),
        "petsc4py": _package_version("petsc4py"),
        "numpy": _package_version("numpy"),
        "argv": sys.argv,
    }

    # MPI rank count, when running under mpiexec.
    try:
        from mpi4py import MPI
        info["mpi_size"] = MPI.COMM_WORLD.size
    except Exception:
        info["mpi_size"] = 1

    # The firedrake/mesh.py int32 patch is a modification to an installed
    # dependency and vanishes silently on reinstall -- record whether it is on.
    try:
        import firedrake.mesh as fdmesh
        src = Path(fdmesh.__file__).read_text()
        info["mesh_int32_patch"] = 'cells_ignore = np.full((npoints, 1), -1, dtype=np.int32' in src
    except Exception:
        info["mesh_int32_patch"] = None

    if extra:
        info.update(extra)
    return info


def _config_hash(config):
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:6]


class Run:
    """A single run directory with config, provenance, metrics, and logs."""

    def __init__(self, slug, config, seed=None, root=None, allow_dirty=False):
        self.slug = slug
        self.config = dict(config)
        if seed is not None:
            self.config["seed"] = seed
        self.seed = seed

        prov = provenance()
        if prov["git_dirty"] and not allow_dirty:
            raise RuntimeError(
                "refusing to launch from a dirty git tree -- a result you cannot tie "
                "to a commit is a result you cannot reproduce. Commit, or pass "
                "allow_dirty=True for a scratch run."
            )

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        name = f"{stamp}_{slug}_{_config_hash(self.config)}"

        base = Path(root) if root else repo_root() / "runs"
        self.dir = base / name
        (self.dir / "output").mkdir(parents=True, exist_ok=True)

        (self.dir / "config.json").write_text(json.dumps(self.config, indent=2, default=str))
        (self.dir / "env.json").write_text(json.dumps(prov, indent=2, default=str))

        self.provenance = prov
        self.rows = []
        self._t0 = time.time()
        self._metrics_path = self.dir / "metrics.csv"

    @property
    def output(self):
        """Directory for VTK and checkpoints. Never leaves the server."""
        return self.dir / "output"

    def record(self, **fields):
        """Append one metrics row. Written through immediately."""
        self.rows.append(dict(fields))
        self._flush()

    def _flush(self):
        if not self.rows:
            return
        keys = list(dict.fromkeys(k for row in self.rows for k in row))
        with self._metrics_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)

    def _peak_rss_gb(self):
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports kB, macOS reports bytes.
            scale = 1024 ** 2 if sys.platform == "darwin" else 1024
            return peak / scale / 1024
        except Exception:
            return None

    def finish(self, status="ok"):
        self._flush()
        summary = {
            "run": self.dir.name,
            "slug": self.slug,
            "seed": self.seed,
            "status": status,
            "wall_s": round(time.time() - self._t0, 2),
            "peak_rss_gb": self._peak_rss_gb(),
            "git_sha": self.provenance["git_sha"],
            "rows": len(self.rows),
        }
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))

        # One appended line in the global index -- this is what you grep later.
        index = self.dir.parent / "index.csv"
        exists = index.exists()
        with index.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)

        return summary

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish(status="ok" if exc_type is None else f"failed:{exc_type.__name__}")
        return False
