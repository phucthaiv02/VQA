import os
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data.datasets import VQADataset, Tokenizer, TextData
from data.utils import load_tokenizer, save_tokenizer

TRAIN_PATH = 'data\\raw\\data_train.csv'
TEST_PATH = 'data\\raw\\data_eval.csv'
IMAGE_FOLDER = 'data\\raw\\images'

TOKENIZER_PKL = 'data\\tokenizer.pkl'
MAX_LEN = 64
BATCH_SIZE = 32

if __name__ == '__main__':
    try:
        tokenizer = load_tokenizer(TOKENIZER_PKL)
    except:
        tokenizer = Tokenizer(max_len=MAX_LEN)
        text_data = TextData(TRAIN_PATH)
        tokenizer.build_vocab(text_data.data)
        save_tokenizer(tokenizer=tokenizer, target=TOKENIZER_PKL)


    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    train_ds = VQADataset(TRAIN_PATH, IMAGE_FOLDER,
                          tokenizer=tokenizer, max_len=MAX_LEN, transform=transform)
    test_ds = VQADataset(TEST_PATH, IMAGE_FOLDER,
                         tokenizer=tokenizer, max_len=MAX_LEN, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    for batch in train_loader:
        print(batch["question"].shape)
        break
