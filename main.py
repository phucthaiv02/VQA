import os
import logging
import numpy as np
import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data.datasets import VQADataset, Tokenizer, TextData
from data.utils import load_tokenizer, save_tokenizer

from models.san import SANModel

from utils import train, evaluate, save_result

parser = argparse.ArgumentParser('Visual Question Answering CLI')
parser.add_argument('--train', action='store_true', help='Train model')
parser.add_argument('--train_data', type=str, help='Path to training data')
parser.add_argument('--eval_data', type=str, help='Path to evaluation data')
parser.add_argument('--image_dir', type=str, help='Path to image folder')
parser.add_argument('--output', type=str)

parser.add_argument('--infer', action='store_true',
                    help='Inference with your data')
parser.add_argument('--model', help='Path to your model')
parser.add_argument('--image', type=str, help='Path to your image')
parser.add_argument('--question', type=str, help='Question about your image')

args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

TOKENIZER_PKL = 'data\\tokenizer.pkl'
MAX_LEN = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10

if __name__ == '__main__':
    if args.train:
        TRAIN_PATH = args.train_data
        TEST_PATH = args.eval_data
        IMAGE_FOLDER = args.image_dir
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

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = SANModel(tokenizer.vocab_size).to('cuda')

    logger.info('Train model ...')
    for epoch in range(1, NUM_EPOCHS + 1):
        model, history = train(model, train_loader)
        # score = evaluate(model, test_loader)
        # saved = save_result(model, history, score)
    elif args.infer:
        if not args.image or not args.question:
            parser.error('Missing --image or ---question argument')
