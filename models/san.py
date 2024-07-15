import torch
import torchvision
from torch import nn


IMAGE_WEIGHTS = 'models\\weights\\vgg19_bn_weights.pth'


class ImageModel(nn.Module):
    def __init__(self, embed_size=1024):
        super(ImageModel, self).__init__()
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
    def __init__(self, vocab_size, word_embed_size, embed_size,  hidden_size, num_layers=2):
        super(QuestionModel, self).__init__()
        self.word2vec = nn.Embedding(vocab_size, word_embed_size)
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
    def __init__(self, vocab_size, embed_size=1024, word_embed_size=500, num_layers=2, hidden_size=64):
        super(SANModel, self).__init__()
        self.image_model = ImageModel(embed_size=embed_size)
        self.question_model = QuestionModel(
            vocab_size=vocab_size, word_embed_size=word_embed_size, embed_size=embed_size, hidden_size=hidden_size)
        self.attention_stack = nn.ModuleList([Attention(512, embed_size)]*2)
        self.mlp = nn.Sequential(nn.Dropout(
            p=0.5), nn.Linear(embed_size, vocab_size), nn.Softmax(dim=1))

    def forward(self, img, qst):
        img_features = self.image_model(img)
        qst_features = self.question_model(qst)
        vi = img_features
        u = qst_features
        for attn_layer in self.attention_stack:
            u = attn_layer(vi, u)

        output = self.mlp(u)
        return output


class Attention(nn.Module):
    def __init__(self, num_channels, embed_size, dropout=True):
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(embed_size, num_channels)
        self.ff_question = nn.Linear(embed_size, num_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(num_channels, 1)

    def forward(self, vi, vq):
        hi = self.ff_image(vi)
        hq = self.ff_question(vq).unsqueeze(dim=1)
        ha = torch.tanh(hi + hq)
        if self.dropout:
            ha = self.dropout(ha)
        hA_atn = self.ff_attention(ha)
        pI = torch.softmax(hA_atn, dim=1)
        vI_attended = (pI * vi).sum(dim=1)
        u = vI_attended + vq

        return u
