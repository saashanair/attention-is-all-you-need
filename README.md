# Attention Is All You Need

A from-scratch PyTorch implementation of the Transformer architecture from
[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), built as a learning exercise. The
encoder/decoder stack, multi-head attention, positional encoding, and feed-forward blocks are all
implemented by hand rather than pulled from `torch.nn.Transformer`. The model trains on the
[Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) dataset for English→German
translation.

The code itself is written and debugged by hand — AI was used only for scaffolding and review, not
for writing the implementation. See [AI usage](#ai-usage) below for exactly how.

## Project layout

```
src/transformer/
├── model/          # Encoder, Decoder, Transformer, and their sub-modules (attention, FFN, etc.)
├── data/           # Dataset loading, preparation, and batching
├── tokenizers/     # Char-level and BPE tokenizer implementations
├── training/       # Training loop, checkpointing, config
├── inference/      # Inference-time generation (in progress)
├── device.py       # Device selection (CPU/CUDA/MPS)
├── paths.py        # Experiment/run path helpers
└── main.py         # Training entrypoint

tests/                # pytest suite mirroring the src/ layout
raw/                  # Cached dataset downloads (gitignored)
```

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

Train the model:

```bash
uv run attention-is-all-you-need
```

Training config (model size, tokenizer, batch size, epochs, resume path, early stopping, etc.) is
set in `src/transformer/main.py`. Runs are checkpointed under `runs/`, with config and tokenizer
state persisted alongside model weights so a run can be resumed.

Run tests and checks:

```bash
uv run pytest
uv run pyright
uv run pre-commit run --all-files
```

## Status

Working:
- Training end-to-end — checkpointing, resume, early stopping, TensorBoard logging.

Not yet done:
- Inference/generation — `predict_next_token` in `model/transformer.py` is a stub; there's no
  decoding loop yet.
- Tests for `data/`, `training/`, and most of `tokenizers/` (`base.py`, `bpe.py`, `prepare.py`) —
  only `model/` and `tokenizers/char.py` are covered so far.
- BPE tokenizer (`tokenizers/bpe.py`) currently wraps the Hugging Face `tokenizers` library rather
  than being implemented from scratch.

## AI usage

This started as a side-quest, pet project to reimplement the Transformer straight from the paper.
The constraint I set myself: AI wasn't allowed to write the code, or hand me code in conversation
— no snippets, no full solutions. That doesn't mean AI played no part at all. I used it to support
the project in a few specific ways:

- **Scaffolding**: setting up the CI workflow (`.github/workflows/ci.yml`), Dependabot config
  (`.github/dependabot.yml`), and pre-commit hooks (`.pre-commit-config.yaml`).
- **Git activity**: running `git commit`/`git push` on my behalf and writing the commit messages
  for changes I make.
- **This README**: written by AI, from the project's actual structure and config.
- **Rubber ducking**: talking through design decisions and bugs, on every `.py` file I write.
- **Code review**: reviewing each `.py` file I write, and flagging test cases I hadn't thought to
  cover on my first pass.

The architecture, training logic, and all the actual problem-solving are mine.
