import torch
from HRR.with_pytorch import cosine_similarity


class VTB:
    def __init__(self, batch_size, d_model, device):
        self.d_model = d_model
        self.device = device
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
        bind = torch.zeros(x.size()).to(self.device)
        size = (x.size()[0], self.d_model, self.d_model)
        for i in range(ch):
            bind[:, i, :, :] = self.bind_single_dim(x[:, i, :, :].flatten(1), y[:, i, :, :].flatten(1)).reshape(*size)
        return bind

    def unbinding(self, x, y, ch=1):
        unbind = torch.zeros(x.size()).to(self.device)
        size = (x.size()[0], self.d_model, self.d_model)
        for i in range(ch):
            unbind[:, i, :, :] = self.unbind_single_dim(x[:, i, :, :].flatten(1), y[:, i, :, :].flatten(1)).reshape(
                *size)
        return unbind / unbind.shape[-1]


class Orthogonal:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def tensor(self, size):
        random = torch.normal(mean=self.mean, std=self.std, size=size)
        q, _ = torch.linalg.qr(random)
        return q

    @staticmethod
    def is_orthogonal(x):
        dim = list(range(len(x.size())))
        dim[-2], dim[-1] = dim[-1], dim[-2]
        x = torch.matmul(x.permute(dim), x)
        x = torch.diagonal(x, dim1=-2, dim2=-1)
        x = torch.sum(x, dim=-1) / x.size()[-1]
        return x


def vtb(batch_size, input_dim, device):
    sampler = Orthogonal(mean=0., std=1. / input_dim)
    module = VTB(batch_size=batch_size, d_model=input_dim, device=device)
    return sampler, module


def normal(shape, std):
    return torch.normal(0., std, shape)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sampler_, module_ = vtb(batch_size=10, input_dim=32)
    x = sampler_.tensor((10, 3, 32, 32)).to(device)
    y = sampler_.tensor((10, 3, 32, 32)).to(device)

    b = module_.binding(x, y, ch=3)
    x_hat = module_.unbinding(b, y, ch=3)

    print("similarity :\n\t", cosine_similarity(x, x_hat, dim=(-2, -1)))
