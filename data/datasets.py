import os
import re
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from collections import defaultdict


class Tokenizer():
    def __init__(self, max_len=1024):
        self.word2idx = {}
        self.idx2word = []
        self.vocab_size = 0
        self.max_len = max_len

    def build_vocab(self, texts):
        word_freq = defaultdict(int)
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_freq[token] += 1

        self.idx2word = ["<PAD>", "<UNK>"] + \
            sorted(word_freq.keys(), key=lambda x: -word_freq[x])
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        self.vocab_size = len(self.word2idx)

    def tokenize(self, text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def encode(self, text, padding=True):
        tokens = self.tokenize(text)
        token_ids = [self.word2idx.get(
            token, self.word2idx["<UNK>"]) for token in tokens]

        if padding:
            token_ids = token_ids[:self.max_len] + \
                [self.word2idx["<PAD>"]] * (self.max_len - len(token_ids))

        return token_ids

    def decode(self, token_ids):
        tokens = [self.idx2word[token_id] for token_id in token_ids]
        return " ".join(tokens)


class VQADataset(Dataset):
    def __init__(self, qa_csv_file, image_folder, tokenizer, max_len, transform=None):
        self.data = pd.read_csv(qa_csv_file)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        # Build vocabulary
        self.tokenizer.build_vocab(
            self.data['question'].tolist() + self.data['answer'].tolist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx, 0]
        answer = self.data.iloc[idx, 1]
        image_id = self.data.iloc[idx, 2]

        # Preprocess image
        image_path = os.path.join(self.image_folder, f'{image_id}.png')
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Preprocess question
        input_ids = self.tokenizer.encode(question, self.max_len)
        question = torch.tensor(input_ids, dtype=torch.long)

        # Preprocess answer
        answer = answer.replace(" ", "").split(",")[0]
        answer = self.tokenizer.encode(answer, padding=False)

        return {
            'image': image,
            'question': question,
            'answer': answer
        }
