import os
import logging
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data.datasets import VQADataset, Tokenizer, TextData
from data.utils import load_tokenizer, save_tokenizer

from models.san import SANModel

from utils import train, evaluate, save_result

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

TRAIN_PATH = 'data\\raw\\data_train.csv'
TEST_PATH = 'data\\raw\\data_eval.csv'
IMAGE_FOLDER = 'data\\raw\\images'

TOKENIZER_PKL = 'data\\tokenizer.pkl'
MAX_LEN = 64
BATCH_SIZE = 1
NUM_EPOCHS = 10

if __name__ == '__main__':
    try:
        logger.info('Load tokenizer ...')
        tokenizer = load_tokenizer(TOKENIZER_PKL)
    except:
        logger.info('Build tokenizer ...')
        tokenizer = Tokenizer(max_len=MAX_LEN)
        text_data = TextData(TRAIN_PATH)
        tokenizer.build_vocab(text_data.data)
        save_tokenizer(tokenizer=tokenizer, target=TOKENIZER_PKL)
        logger.info(f'Saved tokenizer to {TOKENIZER_PKL}')

    logger.info(f'Vocab size: {tokenizer.vocab_size}')

    logger.info('Load dataset ...')
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

    model = SANModel(tokenizer.vocab_size).to('cuda')

    logger.info('Train model ...')
    for epoch in range(1, NUM_EPOCHS + 1):
        model, history = train(model, train_loader)
        # score = evaluate(model, test_loader)
        # saved = save_result(model, history, score)
