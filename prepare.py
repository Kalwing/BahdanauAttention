import random
import re
import torch
from config import (SEED, MAX_LENGTH, N_WORDS, TRAIN_SPLIT, VAL_SPLIT,
                    BASE_PATH, IN_DATA_NAME, OUT_DATA_NAME)


# Modules
if SEED is not None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.word2index = {self.index2word[idx]: idx for idx in self.index2word.keys()}
        self.word2count = {values: 0 for values in self.index2word.values()}
        self.n_words = 4  # next key of index2word which already contains 3 words (SOS,EOS,UNK)

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        try:
            self.word2count[word] += 1 # nb of occurrences of the word in the text
        except KeyError:
            self.word2index[word] = self.n_words # add word in word2index
            self.word2count[word] = 1
            self.index2word[self.n_words] = word # add word in index2word
            self.n_words += 1 # next key


def prepare_line(line, common_words):
    line = line.lower().replace("...", "DOT")
    line = re.sub('[^A-Za-z0-9 ]+', '', line)
    return " ".join([
            word if word in common_words else 'UNK'
        for word in line.split()
    ])


def readLang(lang_file):
    print("Reading lines...")

    # Read the file and split into lines
    with open(str(BASE_PATH/lang_file), "r", encoding='utf-8') as fin:
        lines = fin.readlines()

    print("Calculating the word distribution...")
    words2count = {}
    for line in lines:
        for word in line.split():
            try:
                words2count[word] += 1
            except KeyError:
                words2count[word] = 1
    words = list(set(words2count.keys()))
    words.sort(key= lambda w: words2count[w], reverse=True)
    common_words = set(words[:N_WORDS])

    print("Selecting only the {} most common words".format(N_WORDS))
    for i in range(len(lines)):
        lines[i] = prepare_line(lines[i], common_words)
    return lines, common_words


def prepareData(input_lang_file, output_lang_file):
    input_lines, words = readLang(input_lang_file)
    output_lines, words = readLang(output_lang_file)

    input_lang = Lang(input_lang_file)
    output_lang = Lang(output_lang_file)

    pairs = list(zip(input_lines, output_lines))
    print("Had %s sentence pairs" % len(pairs))
    pairs = [(pair[0], pair[1]) for pair in pairs if len(pair[0].split()) <= MAX_LENGTH]
    print(pairs[:5])
    print("Kept %s sentence pairs after removing the one too long" % len(pairs))

    print("\nCounting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print("Counted words (difference with n most common words come from trimming):")
    print(output_lang.name, output_lang.n_words)

    print("Number of UNK: {}".format(output_lang.word2count['UNK']))
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    """
    tokenizarion : replaces the word with its index in the dictionary

    :param lang: The language of the sentence
    :type lang: Lang
    :param sentence: A sentence
    :type sentence: str

    :return: An array of indexes corresponding to word in sentence
    """
    return [lang.word2index[word] for word in sentence.split(' ')]


def sentenceFromIndexes(lang, sentence):
    """
    tokenizarion : replaces the word with its index in the dictionary

    :param lang: The language of the sentence
    :type lang: Lang
    :param sentence: A sentence made of a serie of tokens
    :type sentence: Tensor

    :return: A string
    """
    return " ".join([lang.index2word[int(word)] for word in sentence])


def tensorFromSentence(lang, sentence):
    indexes = [SOS_token]
    indexes.extend(indexesFromSentence(lang, sentence))
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


if OUT_DATA_NAME is None:
    input_lang, output_lang, pairs = prepareData(str(IN_DATA_NAME), str(IN_DATA_NAME))
else:
    input_lang, output_lang, pairs = prepareData(str(IN_DATA_NAME), str(OUT_DATA_NAME))

n = len(pairs)

ntrain = int(TRAIN_SPLIT*n)
nvalid = int(VAL_SPLIT*n)
ntest = n - ntrain - nvalid

idx = list(range(len(pairs)))
random.shuffle(idx)
shuffle_pairs = [tensorsFromPair(pairs[i]) for i in idx]

train_pairs = shuffle_pairs[0:ntrain]
valid_pairs = shuffle_pairs[ntrain:ntrain+nvalid]
test_pairs = shuffle_pairs[ntrain+nvalid:]

train_pairs[0][0], output_lang.index2word[614]
