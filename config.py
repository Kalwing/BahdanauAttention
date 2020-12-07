from pathlib import Path
import os

MAX_LENGTH = 30
N_WORDS = 60000

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
# test percentage is the rest
assert TRAIN_SPLIT + VAL_SPLIT <= 1

BASE_PATH = Path("DATASETS")
os.makedirs(str(BASE_PATH), exist_ok=True)
SAVE_FOLDER = Path("RESULTS")
os.makedirs(str(SAVE_FOLDER), exist_ok=True)

FINETUNE_MODEL = None #"07-12-model.pt"

LANG_REF_FILE = None #"eng_train3.txt"
IN_DATA_NAME = "eng_train3.txt"
OUT_DATA_NAME = None #"yoda_out.txt"

BATCH_SIZE = 256
N_EPOCHS = 13
CLIP = 1 # Clip gradients norm to this value

SEED = 42