import torch
import torch.nn as nn
from utils import normal_mixture, index_sequence, cosine_similarity


class Network(nn.Module):
    def __init__(self, device, in_features, hidden, out_features, labels, requires_grad=True, negative=False,
                 factor=1, drop_rate=0., scale=0.5, reduce_dim=False, kernel_dim=None):
        super().__init__()

        self.pos = nn.Parameter(normal_mixture((1, out_features), scale=scale), requires_grad=requires_grad)
        self.neg = nn.Parameter(normal_mixture((1, out_features), scale=scale), requires_grad=requires_grad)
        self.cls = nn.Parameter(
            torch.concat([normal_mixture((labels, out_features), scale=scale),
                          torch.zeros((1, out_features))], dim=0), requires_grad=requires_grad
        )
        self.all_cls = torch.sum(torch.unsqueeze(self.cls, dim=0), dim=1).to(device)

        self.scale = scale
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

        cp = logits / self.pos
        pos = torch.sum(self.cls[true], dim=1)
        with torch.no_grad():
            rho = (torch.norm(logits, p=2, dim=-1) ** 2 / ((self.scale ** 4) * logits.shape[-1])).clip(1.0)
        cosine = cosine_similarity(pos, cp, dim=-1)
        augment = torch.where(cosine > 0, cosine * torch.sqrt(rho), 0).clip(0, 1)
        jp = torch.mean(1 - augment)

        jn = 0.
        if self.negative:
            cn = logits / self.neg
            neg = self.all_cls.unsqueeze(0).repeat(b, 1, 1) - pos
            augment = torch.clamp(cosine_similarity(neg, cn, dim=-1) * torch.sqrt(rho), 0, 1)
            jn = torch.mean(augment)
        return jp + jn

    def inference(self, logits, steps=1):
        unbind = logits / self.pos
        unbind = torch.unsqueeze(unbind, dim=1)
        cls = torch.unsqueeze(self.cls, dim=0)
        score = []
        for (i, j) in index_sequence(self.labels // steps, self.labels):
            score.append(cosine_similarity(unbind, cls[:, i:j, :], dim=-1))
        return torch.abs(torch.concatenate(score, dim=1))
