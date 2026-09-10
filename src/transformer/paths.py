import functools
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for parent in start.parents:
        if (parent / 'pyproject.toml').is_file():
            return parent
    raise RuntimeError('project root not found')
