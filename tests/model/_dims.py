N_ENC = 2
N_DEC = 3
D_MODEL = 4
D_FF = 16
HEADS = 2
D_KQ = D_V = D_MODEL // HEADS
P_DROP = 0.1

BATCH = 3
MAX_SEQ_LEN = 10

# for tests that don't care about src/tgt/q/kv
VOCAB_SIZE = 15
SEQ_LEN = 6

# for tests that need an src / tgt distinction
VOCAB_SIZE_SRC = 15
SL_SRC = 6

VOCAB_SIZE_TGT = 20
SL_TGT = 5

# for attention-specific tests
SL_Q = 5
SL_KV = 10
