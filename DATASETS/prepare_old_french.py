import os
import numpy as np
from pathlib import Path
import unicodedata
import re


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s).strip()
    if len(s) > 2:
        return s
    else:
        return ''


base_path = Path("Project/DATASETS/dataset_pretrain_oldfrench")

N = 0
with open(base_path/"dataset.gt", 'w') as fdata:
    all_s = []
    for f in base_path.iterdir():
        if f.suffix != '.txt':
            continue
        print(f)
        with open(f, 'r') as fin:
            txt = fin.readlines()
        # TODO: Retour a la ligne should't end sentence
        # txt = " ".join("".join(txt).split('\n'))
        SPLIT_CHARS = ("\n",".",":","!","?",)
        clean_sentences = txt
        for c in SPLIT_CHARS:
            sentences = clean_sentences
            clean_sentences = []
            for s in sentences:
                clean_sentences.extend(s.split(c))
        print(f"\tSeparated {len(clean_sentences)} sentences")

        sentences = [normalizeString(sentence) for sentence in sentences]
        sentences = [sentence + '\n' for sentence in sentences if len(sentence) > 1]
        N += len(sentences)
        all_s.extend(sentences)
    print(f"Dataset of {N} sentences")

    print("Removing pairs...")
    all_s = list(set(all_s))

    print("Sorting by length...")
    all_s.sort(key=lambda s: len(s))

    fdata.writelines(all_s)

