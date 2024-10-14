import torch
from math import sqrt, isqrt
from HRR.with_pytorch import cosine_similarity
from torch.nn.functional import pad


class VTB:
    def __init__(self, batch_size, d_model):
        self.d_model = d_model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask = torch.zeros((batch_size, d_model ** 2, d_model ** 2)).to(self.device)
        self.ones = torch.ones((batch_size, d_model, d_model)).to(self.device)

        for i in range(0, self.d_model * self.d_model, self.d_model):
            self.mask[:, i:i + self.d_model, i:i + self.d_model] += self.ones

    def block_diagonal(self, x, n):
        batch_size = x.size()[0]
        x = torch.tile(x, dims=[n, n])
        x = x * self.mask[0:batch_size]
        return x

    def bind_single_dim(self, x, y):
        d = torch.tensor(x.size()[1])
        d_prime = torch.sqrt(d).int()
        vy_prime = torch.pow(d, 1.0 / 4.0) * torch.reshape(y, (x.shape[0], d_prime, d_prime))
        vy = self.block_diagonal(vy_prime, d_prime)
        return torch.matmul(vy, x.unsqueeze(-1)).squeeze()

    def unbind_single_dim(self, x, y):
        d = torch.tensor(x.size()[1])
        d_prime = torch.sqrt(d).int()
        vy_prime = torch.pow(d, 1.0 / 4.0) * torch.reshape(y, (x.shape[0], d_prime, d_prime))
        vy = self.block_diagonal(vy_prime.permute(0, 2, 1), d_prime)
        return torch.matmul(vy, x.unsqueeze(-1)).squeeze()

    def binding(self, x, y, ch=1):
        org_d = x.shape[-1]
        if x.shape[-1] != self.d_model ** 2:
            p = (isqrt(org_d) + 1) ** 2 - org_d
            x = pad(x, (0, p))
            y = pad(y, (0, p))

        shape = x.shape
        d_prime = torch.sqrt(torch.tensor(shape[-1])).int()
        x = torch.reshape(x, (shape[0], -1, d_prime, d_prime))
        y = torch.reshape(y, (shape[0], -1, d_prime, d_prime))
        bind = torch.zeros(x.shape).to(self.device)
        size = (shape[0], self.d_model, self.d_model)
        for i in range(ch):
            bind[:, i, :, :] = self.bind_single_dim(x[:, i, :, :].flatten(1), y[:, i, :, :].flatten(1)).reshape(*size)
        bind = torch.reshape(bind, (shape[0], -1))
        return bind[:, :org_d]

    def unbinding(self, x, y, ch=1):
        org_d = x.shape[-1]
        if x.shape[-1] != self.d_model ** 2:
            p = (isqrt(org_d) + 1) ** 2 - org_d
            x = pad(x, (0, p))
            y = pad(y, (0, p))

        shape = x.shape
        d_prime = torch.sqrt(torch.tensor(shape[-1])).int()
        x = torch.reshape(x, (x.shape[0], -1, d_prime, d_prime))
        y = torch.reshape(y, (y.shape[0], -1, d_prime, d_prime))
        y = y.repeat(x.shape[0] - y.shape[0] + 1, 1, 1, 1)

        unbind = torch.zeros(x.shape).to(self.device)
        size = (shape[0], self.d_model, self.d_model)
        for i in range(ch):
            unbind[:, i, :, :] = self.unbind_single_dim(x[:, i, :, :].flatten(1), y[:, i, :, :].flatten(1)).reshape(
                *size)
        unbind = unbind / unbind.shape[-1]
        unbind = torch.reshape(unbind, (shape[0], -1))
        return unbind[:, :org_d]


class Orthogonal:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def tensor(self, size):
        org_d = size[-1]
        if org_d != isqrt(org_d) ** 2:
            d_prime = int(sqrt(size[-1])) + 1
        else:
            d_prime = int(sqrt(size[-1]))
        size = (size[0], 1, d_prime, d_prime)
        random = torch.normal(mean=self.mean, std=self.std, size=size)
        # random = torch.distributions.uniform.Uniform(0, 1).sample(size)
        q, _ = torch.linalg.qr(random)
        q = torch.reshape(q, (size[0], d_prime ** 2))
        return q[:, :org_d]

    @staticmethod
    def is_orthogonal(x):
        dim = list(range(len(x.size())))
        dim[-2], dim[-1] = dim[-1], dim[-2]
        x = torch.matmul(x.permute(dim), x)
        x = torch.diagonal(x, dim1=-2, dim2=-1)
        x = torch.sum(x, dim=-1) / x.size()[-1]
        return x


def vtb(batch_size, input_dim):
    if input_dim != isqrt(input_dim) ** 2:
        input_dim = (isqrt(input_dim) + 1) ** 2
    sampler = Orthogonal(mean=0., std=1. / input_dim)
    module = VTB(batch_size=batch_size, d_model=int(sqrt(input_dim)))
    return sampler, module


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    d = 400
    sampler_, module_ = vtb(batch_size=10, input_dim=d)

    x = sampler_.tensor((10, d)).to(device)
    y = sampler_.tensor((10, d)).to(device)

    b = module_.binding(x, y)
    x_hat = module_.unbinding(b, y)

    print(x.shape)
    print(x_hat.shape)
    print("similarity :\n\t", cosine_similarity(x, x_hat, dim=-1))
