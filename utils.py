import torch
from torch import nn

LR = 0.05


def loss_fn(logits, labels):
    logits = logits.squeeze()
    labels = labels.squeeze()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    return loss


def get_optimizer(model, lr=0.005):
    return torch.optim.SGD(model.parameters(), lr=lr)


def train(model, dataloader):
    optimizer = get_optimizer(model)
    model.train()

    batch_loss = []
    for batch in dataloader:
        images = batch['image'].to('cuda')
        questions = batch['question'].to('cuda')
        answer = batch['answer'].to('cuda')

        predict = model(images, questions)
        loss = loss_fn(predict, answer)

        batch_loss.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = sum(batch_loss) / len(dataloader.dataset)
    print('epoch loss: ', avg_loss)


def evaluate():
    pass


def save_result():
    pass


# logits = torch.tensor([1.2, 0.5, -0.3])
# labels = torch.tensor([0])

# criterion = nn.CrossEntropyLoss()

# loss = criterion(logits, labels)
# print(loss.item())
