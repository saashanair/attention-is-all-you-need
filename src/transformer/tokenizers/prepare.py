from pathlib import Path

from datasets import DatasetDict

from . import BPETokenizer, CharTokenizer, TokenizerConfig

TOKENIZER_CLASSES = {
    'char': CharTokenizer,
    'bpe': BPETokenizer,
}


def get_kwargs(corpus, vocab_size):
    tok_kwargs = {}
    tok_kwargs['corpus'] = corpus
    if vocab_size:
        tok_kwargs['vocab_size'] = vocab_size
    return tok_kwargs


def build_tokenizers(
    raw_data: DatasetDict, src_lang: str, tgt_lang: str, tokenizer_cfg: TokenizerConfig, resume: bool, store_path: Path
):
    corpus_src = list(raw_data['train'][src_lang])
    corpus_tgt = list(raw_data['train'][tgt_lang])

    src_tok_save_path = f'{store_path}/src_tok.json'
    tgt_tok_save_path = f'{store_path}/tgt_tok.json'

    if resume:
        return TOKENIZER_CLASSES[tokenizer_cfg.tokenizer_type].load(path=src_tok_save_path), TOKENIZER_CLASSES[
            tokenizer_cfg.tokenizer_type
        ].load(path=tgt_tok_save_path)

    if tokenizer_cfg.shared_tokenizer:
        tok_kwargs = get_kwargs(corpus=corpus_src + corpus_tgt, vocab_size=tokenizer_cfg.vocab_size)
        src_tok = tgt_tok = TOKENIZER_CLASSES[tokenizer_cfg.tokenizer_type](**tok_kwargs)
    else:
        src_tok_kwargs = get_kwargs(corpus=corpus_src, vocab_size=tokenizer_cfg.src_vocab_size)
        tgt_tok_kwargs = get_kwargs(corpus=corpus_tgt, vocab_size=tokenizer_cfg.tgt_vocab_size)

        src_tok = TOKENIZER_CLASSES[tokenizer_cfg.tokenizer_type](**src_tok_kwargs)
        tgt_tok = TOKENIZER_CLASSES[tokenizer_cfg.tokenizer_type](**tgt_tok_kwargs)

    src_tok.save(path=src_tok_save_path)
    tgt_tok.save(path=tgt_tok_save_path)

    return src_tok, tgt_tok
