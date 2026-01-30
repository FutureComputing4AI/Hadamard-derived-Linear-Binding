import torch
import torch.nn as nn
from utils import normal_mixture, cosine_similarity


class Network(nn.Module):
    def __init__(self, in_features, hidden, out_features, labels, requires_grad=True, negative=False,
                 factor=1, drop_rate=0., scale=0.5):
        super().__init__()

        self.pos = nn.Parameter(normal_mixture((1, out_features), scale=scale), requires_grad=requires_grad)
        self.neg = nn.Parameter(normal_mixture((1, out_features), scale=scale), requires_grad=requires_grad)
        self.cls = nn.Parameter(normal_mixture((1, labels, out_features), scale=scale), requires_grad=requires_grad)

        self.scale = scale
        self.negative = negative

        self.network = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden, hidden * factor),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden * factor, out_features),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)

    def loss(self, logits, true):
        mask = torch.unsqueeze(true, dim=-1)

        cp = logits / self.pos
        pos = torch.sum(self.cls * mask, dim=1)
        with torch.no_grad():
            rho = (torch.norm(logits, p=2, dim=-1) ** 2 / ((self.scale ** 4) * logits.shape[-1])).clip(1.0)
        cosine = cosine_similarity(pos, cp, dim=-1)
        augment = torch.where(cosine > 0, cosine * torch.sqrt(rho), 0).clip(0, 1)
        jp = torch.mean(1 - augment)

        jn = 0.
        if self.negative:
            cn = logits / self.neg
            neg = torch.sum(self.cls * (1. - mask), dim=1)
            augment = torch.clamp(cosine_similarity(neg, cn, dim=-1) * torch.sqrt(rho), 0, 1)
            jn = torch.mean(augment)
        return jp + jn

    def inference(self, logits):
        unbind = logits / self.pos
        unbind = torch.unsqueeze(unbind, dim=1)
        return torch.abs(cosine_similarity(unbind, self.cls, dim=-1))
