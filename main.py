import random
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from pathlib import Path
from tqdm import tqdm

from prepare import (input_lang, output_lang, pairs, train_pairs, valid_pairs,
                    test_pairs)
from models import Attention, Encoder, Decoder, Seq2Seq
from utils import loss_holder
from config import SEED, BATCH_SIZE, N_EPOCHS, CLIP


if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dict_dim = len(input_lang.word2index)
output_dict_dim = len(output_lang.word2index)

attn = Attention(hidden_size=1000, hidden_unit=1000)
enc = Encoder(input_dict_dim, hidden_size=1000, embedding_dim=620)
dec = Decoder(output_dict_dim, attn, hidden_size=1000, embedding_dim=620)
model = Seq2Seq(encoder=enc, decoder=dec, device=device).to(device)

# define the optimizer
optimizer = optim.Adam(model.parameters())

# initialize the loss function
criterion = nn.CrossEntropyLoss()



def train(model, optimizer, train_pairs, criterion, clip, BATCH_SIZE):
    model.train()
    nbatch = int(np.floor(len(train_pairs)/BATCH_SIZE))
    epoch_loss = 0

    train_progress = tqdm(range(nbatch-1), ncols=100)
    train_progress.set_description("\033[39m Train")
    statistics_output = loss_holder(train_progress, len_set=nbatch)

    for i in train_progress:
        batch = train_pairs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        src = [batch[k][0] for k in range(BATCH_SIZE)]
        trg = [batch[k][1] for k in range(BATCH_SIZE)]

        # pad all tensors to have same length
        max_len = max([x.numel() for x in src])
        data = [torch.nn.functional.pad(x, pad=(0,0,0,max_len - x.numel()), mode='constant', value=1) for x in src]
        src = torch.stack(data)
        src = src[:,:,0]
        src = src.permute(1,0)

        max_len = max([x.numel() for x in trg])
        data = [torch.nn.functional.pad(x, pad=(0,0,0,max_len - x.numel()), mode='constant', value=1) for x in trg]
        trg = torch.stack(data)
        trg = trg[:,:,0]
        trg = trg.permute(1,0)

        optimizer.zero_grad()

        output = model(src, trg)
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:]
        trg = trg.reshape(trg.shape[0]*trg.shape[1])
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        statistics_output.update(loss)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(nbatch)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def evaluate(model, pairs, criterion):
    nbatch = int(np.floor(len(pairs) / BATCH_SIZE))

    progress = tqdm(range(nbatch-1), ncols=100)
    progress.set_description("\033[90m Val")
    statistics_output = loss_holder(progress, len_set=nbatch)

    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i in progress:
            batch = pairs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            src = [batch[k][0] for k in range(BATCH_SIZE)]
            trg = [batch[k][1] for k in range(BATCH_SIZE)]

            # pad all tensors to have same length
            max_len = max([x.numel() for x in src])
            data = [torch.nn.functional.pad(x, pad=(0,0,0,max_len - x.numel()), mode='constant', value=1) for x in src]
            src = torch.stack(data)
            src = src[:,:,0]
            src = src.permute(1,0)

            max_len = max([x.numel() for x in trg])
            data = [torch.nn.functional.pad(x, pad=(0,0,0,max_len - x.numel()), mode='constant', value=1) for x in trg]
            trg = torch.stack(data)
            trg = trg[:,:,0]
            trg = trg.permute(1,0)

            output = model(src, trg, 0) #turn off teacher forcing for valid

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:]
            trg = trg.reshape(trg.shape[0]*trg.shape[1])

            loss = criterion(output, trg)
            statistics_output.update(loss)
            epoch_loss += loss.item()
    return epoch_loss / len(nbatch)


if __name__ == '__main__':
    print(device)

    print("\nExemples de phrases:")
    for i in range(3):
        pair = random.choice(pairs)
        print("\t«{}» ->\t{}".format(pair[0].strip(), pair[1].strip()))

    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model,optimizer, train_pairs, criterion, CLIP,BATCH_SIZE)
        # vérifier que l'output du décodeur est bien de taille (batch_size, taille du output dico) --> ok !
        valid_loss = evaluate(model, valid_pairs, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut3-model.pt')

        print('Epoch: {} | Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
        print('\tTrain Loss: {:.3f}'.format(train_loss))
        print('\t Val. Loss: {:.3f} '.format(valid_loss))

    valid_loss = evaluate(model, valid_pairs, criterion)
    model.load_state_dict(torch.load('tut3-model.pt'))
    test_loss = evaluate(model, train_pairs, criterion)
    print('| Test Loss: {:.3}'.format(test_loss))
