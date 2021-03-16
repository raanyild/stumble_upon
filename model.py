import torch
import torch.nn as nn

class tncnet(nn.Module):
    def __init__(self, vocab):
        super(tncnet, self).__init__()
        self.text_feature = nn.Sequential(
            nn.Linear(300, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
        self.numeric_feature = nn.Sequential(
            nn.Linear(18, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.embedding = nn.EmbeddingBag.from_pretrained(vocab.vectors)
        
    def forward(self, xt, xo, xn):
        xt = self.embedding(xt, xo)
        xt = self.text_feature(xt)
        xn = self.numeric_feature(xn)
        x = torch.cat((xt, xn), dim=1)
        x = self.classifier(x)
        return x