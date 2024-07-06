import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data.datasets import VQADataset, Tokenizer


TRAIN_PATH = 'data\\raw\\data_train.csv'
TEST_PATH = 'data\\raw\\data_eval.csv'
IMAGE_FOLDER = 'data\\raw\\images'

MAX_LEN = 64
BATCH_SIZE = 32

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    tokenzier = Tokenizer(max_len=MAX_LEN)

    train_ds = VQADataset(TRAIN_PATH, IMAGE_FOLDER,
                          tokenizer=tokenzier, max_len=MAX_LEN, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    for batch in train_loader:
        print(batch['image'].shape)
        print(batch['question'].shape)
        print(batch['answer'])
        break
