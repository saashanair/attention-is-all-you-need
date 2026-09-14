from dataclasses import dataclass
from pathlib import Path

from ..tokenizers import TokenizerConfig, build_tokenizers
from .config import DataConfig
from .dataset import TranslationDataset, build_translation_dataset
from .sources import load_data_from_hf


# not using pydantic here to reduce validation overhead,
# as this relies on code generated data, and doesn't requires the same level of validation as user-input
@dataclass(frozen=True)
class TranslationPipeline:
    data_dict: dict[str, TranslationDataset]
    src_vocab_size: int
    tgt_vocab_size: int
    pad_id: int


def build_translation_pipeline(
    data_cfg: DataConfig, tok_cfg: TokenizerConfig, max_seq_len: int, tok_store_path: Path, resume: bool
) -> TranslationPipeline:
    print('preparing data (download + tokenizer + pre-process / encoding)')
    raw_data = load_data_from_hf(dataset_name=data_cfg.dataset_name)
    src_tok, tgt_tok = build_tokenizers(
        raw_data=raw_data,
        src_lang=data_cfg.src_lang,
        tgt_lang=data_cfg.tgt_lang,
        tokenizer_cfg=tok_cfg,
        store_path=tok_store_path,
        resume=resume,
    )
    data_dict = build_translation_dataset(
        raw=raw_data,
        src_lang=data_cfg.src_lang,
        tgt_lang=data_cfg.tgt_lang,
        max_seq_len=max_seq_len,
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
    )
    return TranslationPipeline(
        data_dict=data_dict, src_vocab_size=src_tok.vocab_size, tgt_vocab_size=tgt_tok.vocab_size, pad_id=src_tok.pad_id
    )
