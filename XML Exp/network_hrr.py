import torch
import torch.nn as nn
from HRR.with_pytorch import unbinding
from utils import index_sequence, cosine_similarity


class Network(nn.Module):
    def __init__(self, in_features, hidden, out_features, labels, requires_grad=True, negative=False,
                 factor=1, drop_rate=0., reduce_dim=False, kernel_dim=None):
        super().__init__()

        self.pos = nn.Parameter(torch.normal(0., 1 / out_features, (1, out_features)), requires_grad=requires_grad)
        self.neg = nn.Parameter(torch.normal(0., 1 / out_features, (1, out_features)), requires_grad=requires_grad)
        self.cls = nn.Parameter(
            torch.concat([torch.normal(0., 1 / out_features, (labels, out_features)),
                          torch.zeros((1, out_features))], dim=0), requires_grad=requires_grad
        )
        self.all_cls = torch.sum(torch.unsqueeze(self.cls, dim=0), dim=1).cuda()

        self.labels = labels + 1
        self.negative = negative
        self.reduce_dim = reduce_dim

        if self.reduce_dim:
            in_features = (in_features - kernel_dim) // kernel_dim + 1

            self.avgpool = nn.Sequential(
                nn.AvgPool1d(kernel_size=kernel_dim, stride=kernel_dim)
            )
            print(f"Reducing dimension with kernel size {kernel_dim}: New input feature size: {in_features}.")

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
        if self.reduce_dim:
            x = self.avgpool(x)

        return self.network(x)

    def loss(self, logits, true):
        b = logits.shape[0]

        cp = unbinding(logits, self.pos, dim=-1)
        pos = torch.sum(self.cls[true], dim=1)
        cosine = torch.abs(cosine_similarity(pos, cp, dim=-1))
        jp = torch.mean(1 - cosine)

        jn = 0.
        if self.negative:
            cn = unbinding(logits, self.neg, dim=-1)
            neg = self.all_cls.unsqueeze(0).repeat(b, 1, 1) - pos
            cosine = torch.abs(cosine_similarity(neg, cn, dim=-1))
            jn = torch.mean(cosine)
        return jp + jn

    def inference(self, logits, steps=1):
        unbind = unbinding(logits, self.pos, dim=-1)
        unbind = torch.unsqueeze(unbind, dim=1)
        cls = torch.unsqueeze(self.cls, dim=0)
        score = []
        for (i, j) in index_sequence(self.labels // steps, self.labels):
            score.append(cosine_similarity(unbind, cls[:, i:j, :], dim=-1))
        return torch.abs(torch.concatenate(score, dim=1))
