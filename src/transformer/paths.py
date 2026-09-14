from pathlib import Path

CHECKPOINTS_DIR = 'checkpoints'
LOGGING_DIR = 'logs'
TOKENIZER_DIR = 'tokenizers'


def project_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for parent in start.parents:
        if (parent / 'pyproject.toml').is_file():
            return parent
    raise RuntimeError('project root not found')


def get_resume_path(resume_path: str | Path) -> Path:
    resume_path = Path(resume_path)
    if not (resume_path.exists() and resume_path.is_dir()):
        raise ValueError(f'resume_path={resume_path} not found')
    if not ((resume_path / CHECKPOINTS_DIR).exists() and len(list((resume_path / CHECKPOINTS_DIR).glob('*.pt'))) > 0):
        raise ValueError(f'no checkpoints found at resume_path={resume_path}')
    if not ((resume_path / TOKENIZER_DIR).exists() and len(list((resume_path / TOKENIZER_DIR).glob('*.json'))) > 0):
        raise ValueError(f'no tokenizer(s) found at resume_path={resume_path}')
    if not ((resume_path / 'config.json').exists()):
        raise ValueError(f'no saved config found to confirm checkpoint compatibility at resume_path={resume_path}')

    (resume_path / LOGGING_DIR).mkdir(parents=True, exist_ok=True)
    return Path(resume_path)


def get_experiment_path(experiment_dir_name: str, resume_path: str | None = None) -> Path:
    if resume_path is not None:
        return get_resume_path(resume_path)

    run_path = Path(f'runs/{experiment_dir_name}')
    chk_dir = run_path / CHECKPOINTS_DIR
    log_dir = run_path / LOGGING_DIR
    tok_dir = run_path / TOKENIZER_DIR

    Path(run_path).mkdir(parents=True, exist_ok=True)
    Path(chk_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(tok_dir).mkdir(parents=True, exist_ok=True)

    return run_path
