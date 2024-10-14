import torch


def index_sequence(batch_size: int, dataset_size: int):
    index_a = list(range(0, dataset_size, batch_size))
    index_b = list(range(batch_size, dataset_size, batch_size))
    index_b.append(dataset_size)
    return list(zip(index_a, index_b))


def normal_mixture(shape, scale=0.5, seed=None):
    if seed:
        torch.manual_seed(seed)
    uniform = torch.rand(shape)
    n1 = torch.normal(-scale, 1 / shape[-1], shape)
    n2 = torch.normal(scale, 1 / shape[-1], shape)
    return torch.where(uniform > 0.5, n1, n2)


def cosine_similarity(x, y, dim=None, keepdim=False):
    if not dim:
        dim = list(range(-len(x.size()) // 2, 0))
    norm_x = torch.norm(x, dim=dim, keepdim=keepdim)
    norm_y = torch.norm(y, dim=dim, keepdim=keepdim)
    return torch.sum(x * y, dim=dim, keepdim=keepdim) / (norm_x * norm_y)


def normalize(val):
    min_val = torch.min(val)
    max_val = torch.max(val)
    return (val - min_val) / (max_val - min_val)


def evaluate(true, pred):
    pred = torch.where(pred > 0.5, 1, 0)
    return torch.mean(torch.all(true == pred, dim=-1).float()) * 100.


def one_hot(labels, n_class):
    y = torch.zeros((labels.shape[0], n_class), dtype=torch.int8)
    y[torch.arange(labels.shape[0])[:, None], labels] = 1
    return y
