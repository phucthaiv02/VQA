import torch
from torch import nn


def loss_fn(logits, labels):
    labels = labels.squeeze()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss = criterion(logits, labels)

    return loss


def get_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)


def train(model, dataloader, learning_rate=5e-3):
    num_samples = len(dataloader.dataset)

    optimizer = get_optimizer(model, learning_rate)
    model.train()

    total_loss = 0
    correct = 0
    for batch in dataloader:
        images = batch['image'].to('cuda')
        questions = batch['question'].to('cuda')
        answer = batch['answer'].to('cuda')

        optimizer.zero_grad()
        output = model(images, questions)
        loss = loss_fn(output, answer)
        loss.backward()
        optimizer.step()

        _, predict = torch.max(output, 1)
        correct += (predict == answer).sum().item()
        total_loss += loss.item()

    avg_loss = total_loss / num_samples
    accuracy = correct / num_samples

    return model, {'loss': avg_loss, 'acc': accuracy}


def evaluate(model, dataloader):
    num_samples = len(dataloader.dataset)

    total_loss = 0
    correct = 0
    model.eval()
    for batch in dataloader:
        images = batch['image'].to('cuda')
        questions = batch['question'].to('cuda')
        answer = batch['answer'].to('cuda')

        output = model(images, questions)
        loss = loss_fn(output, answer)

        _, predict = torch.max(output, 1)

        total_loss += loss.item()
        correct += (predict == answer).sum().item()

    avg_loss = total_loss / num_samples
    accuracy = correct / num_samples

    return {'loss': avg_loss, 'acc': accuracy}


def save_result():
    pass
