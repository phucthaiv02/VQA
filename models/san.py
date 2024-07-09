import torch
import torchvision
from torch import nn


IMAGE_WEIGHTS = 'models\\weights\\vgg19_bn_weights.pth'


class ImageModel(nn.Module):
    def __init__(self, embed_size=500):
        super().__init__()
        try:
            model = torchvision.models.vgg19_bn()
            model.load_state_dict(torch.load(IMAGE_WEIGHTS))
        except:
            model = torchvision.models.vgg19_bn(
                weights=torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1)
            torch.save(model.state_dict(), IMAGE_WEIGHTS)

        vgg_feature = list(model.features.children())
        self.backbone = nn.Sequential(*vgg_feature)
        self.perceptron = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024), nn.Tanh())

    def forward(self, image):
        img_features = self.backbone(image)  # (batch_size, 512, 14, 14)
        img_features = img_features.view(-1, 512, 196).transpose(1, 2)
        img_features = self.perceptron(img_features)

        return img_features  # [batch_size, 196, 512]


class QuestionModel(nn.Module):
    def __init__(self, vocab_size, input_size=500, embed_size=500, hidden_size=1024):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(
            input_size=500, hidden_size=hidden_size, batch_first=True)

    def forward(self, text):
        text_features = self.embed(text)
        output, (h_n, c_n) = self.lstm(text_features)
        return h_n  # ht, (batch_size, embed_size)


class SANModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.image_model = ImageModel()
        self.question_model = QuestionModel(vocab_size)

        self.tanh = nn.Tanh()

        self.ff_question = nn.Linear(in_features=1024, out_features=512)
        self.ff_image = nn.Linear(
            in_features=1024, out_features=512, bias=False)
        self.ff_attention = nn.Linear(in_features=512, out_features=1)

        self.ff_ans = nn.Linear(in_features=1024, out_features=vocab_size)

        self.softmax = nn.Softmax(dim=1)

    def compute_attention(self, vi, vq):
        hA = self.tanh(self.ff_image(vi) + self.ff_question(vq))
        hA_atn = self.ff_attention(hA).squeeze(dim=2)
        pI = self.softmax(hA_atn)
        vI_attended = (pI.unsqueeze(dim=2) * vi)
        vI_attended = vI_attended.sum(dim=1)
        u = vI_attended + vq.squeeze(dim=1)
        return u

    def forward(self, image, question):
        vI = self.image_model(image)  # (batch_size, 196, 1024)
        vQ = self.question_model(question)  # (batch_size, 1, 1024)

        u0 = self.compute_attention(vI, vQ)

        u1 = self.compute_attention(vI, u0)
        ff_u1 = self.ff_ans(u1)
        p_ans = self.softmax(ff_u1)
        return p_ans
