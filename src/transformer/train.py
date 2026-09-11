from functools import partial
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pydantic import BaseModel, PositiveFloat, PositiveInt, Field, model_validator
from .model import TransformerArchConfig, TransformerVocabConfig, Transformer
from .data import load_data_from_hf, build_translation_dataset, collate_translation_batch
from .tokenizers import CharTokenizer

class OptimiserConfig(BaseModel):
    beta1: PositiveFloat = Field(default=0.9, lt=1.0)
    beta2: PositiveFloat = Field(default=0.98, lt=1.0)
    eps: PositiveFloat = 1e-9
    warmup_steps: PositiveInt = 4000

class TrainConfig(BaseModel):
    transformer_arch_cfg: TransformerArchConfig
    optim_cfg: OptimiserConfig
    dataset_name: str = 'bentrevett/multi30k'
    src_lang: str = 'en'
    tgt_lang: str = 'de'
    num_epochs: PositiveInt = 2
    batch_size: PositiveInt = 32
    label_smoothing: float = Field(default=0.1, ge=0.0, lt=1.0)
    shared_tokenizer: bool = True
    shared_embeddings: bool = True

    @model_validator(mode='after')
    def _check_embedding_sharing(self):
        if self.shared_embeddings and not self.shared_tokenizer:
            raise ValueError('shared_embeddings=True requires shared_tokenizer=True')
        return self

def get_lr(step: int, d_model: int, warmup_steps: int) -> float:
    step = max(step, 1)
    return d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)

def prepare_data(dataset_name, src_lang, tgt_lang, max_seq_len, shared_tokenizer):
    print('preparing data (download + tokenizer + pre-process / encoding)')
    raw_data = load_data_from_hf(dataset_name=dataset_name)

    if shared_tokenizer:
        tok_corpus = list(raw_data['train'][src_lang]) + list(raw_data['train'][tgt_lang])
        src_tok = tgt_tok = CharTokenizer(corpus=tok_corpus)
    else:
        src_tok = CharTokenizer(corpus=list(raw_data['train'][src_lang]))
        tgt_tok = CharTokenizer(corpus=list(raw_data['train'][tgt_lang]))

    data_dict = build_translation_dataset(raw=raw_data, src_lang=src_lang, tgt_lang=tgt_lang, max_seq_len=max_seq_len, src_tokenizer=src_tok, tgt_tokenizer=tgt_tok)
    return data_dict, src_tok.vocab_size, tgt_tok.vocab_size, src_tok.pad_id

def train(cfg: TrainConfig):
    ta_cfg=cfg.transformer_arch_cfg
    optim_cfg = cfg.optim_cfg

    data, src_vocab_size, tgt_vocab_size, pad_id = prepare_data(dataset_name=cfg.dataset_name, src_lang=cfg.src_lang, tgt_lang=cfg.tgt_lang, max_seq_len=ta_cfg.max_seq_len, shared_tokenizer=cfg.shared_tokenizer)
    train_loader = DataLoader(dataset=data['train'], batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=partial(collate_translation_batch, pad_id=pad_id))
    val_loader = DataLoader(dataset=data['validation'], batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=partial(collate_translation_batch, pad_id=pad_id))
    test_loader = DataLoader(dataset=data['test'], batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=partial(collate_translation_batch, pad_id=pad_id))

    tv_cfg = TransformerVocabConfig(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, shared_embeddings=cfg.shared_embeddings)

    transformer = Transformer(arch_cfg=ta_cfg, vocab_cfg=tv_cfg)

    # we are using LambdaLR to implement the lr_rate function defined in the paper
    # LambdaLR works as: lr = base_lr * lr_lambda(step_num), where base_lr is the frozen inital lr; thus setting lr to 1.0 ensure we replicate the function as defined in the paper
    optim = torch.optim.Adam(params=transformer.parameters(), lr=1.0, betas=(optim_cfg.beta1, optim_cfg.beta2), eps=optim_cfg.eps)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda step_num: get_lr(step_num, ta_cfg.d_model, optim_cfg.warmup_steps))

    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing, ignore_index=pad_id)

    for epoch in range(cfg.num_epochs):
        transformer.train()

        batch_train_loss = 0
        batch_val_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            logits = transformer(src=batch['encoder_input'], tgt=batch['decoder_input'], src_pad_mask=batch['src_pad_mask'], tgt_pad_mask=batch['tgt_pad_mask'])
            
            logits = torch.reshape(logits, (-1, logits.size(-1))) # (batch * sequence_length, vocab_size)
            labels = torch.reshape(batch['label'], (-1,)) # (batch * sequence_length,)

            loss = loss_fn(logits, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_scheduler.step()

            batch_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

        transformer.eval()
        with torch.no_grad():
            for vbatch_idx, vbatch in enumerate(tqdm(val_loader, desc="  val", unit="batch", leave=False)):
                vout = transformer(src=vbatch['encoder_input'], tgt=vbatch['decoder_input'], src_pad_mask=vbatch['src_pad_mask'], tgt_pad_mask=vbatch['tgt_pad_mask'])

                vout = torch.reshape(vout, (-1, vout.size(-1)))
                vlabels = torch.reshape(vbatch['label'], (-1,))

                vloss = loss_fn(vout, vlabels)
                batch_val_loss += vloss.item()

        ### LOGGING
        #print(f'Epoch: {epoch+1:03d}/{cfg.num_epochs:03d}'
        #    f' | Train/Val Loss: {batch_train_loss / (batch_idx+1):.2f} / {batch_val_loss / (vbatch_idx+1):.2f}')
        tqdm.write(
            f"Epoch {epoch+1:03d}/{cfg.num_epochs:03d}"
            f" | train loss {batch_train_loss / (batch_idx + 1):.3f}"
            f" | val loss {batch_val_loss / (vbatch_idx + 1):.3f}"
        )

    


