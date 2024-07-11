import torch
import torchvision
from torch import nn


IMAGE_WEIGHTS = 'models\\weights\\vgg19_bn_weights.pth'


class ImageModel(nn.Module):
    def __init__(self, embed_size=1024):
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
            nn.Linear(in_features=512, out_features=embed_size), nn.Tanh())

    def forward(self, image):
        with torch.no_grad():
            img_features = self.backbone(image)     # (batch_size, 512, 14, 14)
        img_features = img_features.view(-1, 512, 196).transpose(1, 2)
        img_features = self.perceptron(img_features)

        return img_features  # [batch_size, 196, 512]


class QuestionModel(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size=500, embed_size=1024, num_layers=2, hidden_size=64):

        super(QuestionModel, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        # 2 for hidden and cell states
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)

    def forward(self, question):
        qst_vec = self.word2vec(question)
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)
        _, (hidden, cell) = self.lstm(qst_vec)
        qst_feature = torch.cat((hidden, cell), 2)
        qst_feature = qst_feature.transpose(0, 1)
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)

        return qst_feature


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
        hi = self.ff_image(vi)
        hq = self.ff_question(vq).unsqueeze(dim=1)
        hA = self.tanh(hi + hq)
        hA_atn = self.ff_attention(hA)
        pI = self.softmax(hA_atn)
        vI_attended = (pI * vi).sum(dim=1)
        u = vI_attended + vq
        return u

    def forward(self, image, question):
        vI = self.image_model(image)  # (batch_size, 196, 1024)
        vQ = self.question_model(question)  # (batch_size, 1, 1024)

        u0 = self.compute_attention(vI, vQ)

        u1 = self.compute_attention(vI, u0)
        ff_u1 = self.ff_ans(u1)
        softmax = nn.Softmax(dim=1)
        p_ans = softmax(ff_u1)
        return p_ans
