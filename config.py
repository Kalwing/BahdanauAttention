from pathlib import Path

MAX_LENGTH = 30
N_WORDS = 60000

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
# test percentage is the rest
assert TRAIN_SPLIT + VAL_SPLIT <= 1

BASE_PATH = Path("DATASETS").resolve()
DATA_NAME = "eng_train4.txt"

BATCH_SIZE = 256
N_EPOCHS = 10
CLIP = 1

SEED = 1234