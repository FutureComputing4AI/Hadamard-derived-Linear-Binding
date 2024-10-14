import torch
import sklearn.metrics as metrics


def mean(x):
    return sum(x) / len(x)


def uniform(shape):
    return -2 * torch.rand(shape) + 1


def normal_mixture(shape, std, scale=0.5, seed=None):
    if seed:
        torch.manual_seed(seed)
    uniform = torch.rand(shape)
    n1 = torch.normal(-scale, std, shape)
    n2 = torch.normal(scale, std, shape)
    return torch.where(uniform > 0.5, n1, n2)


def normalize(val):
    min_val = torch.min(val)
    max_val = torch.max(val)
    return (val - min_val) / (max_val - min_val)


def evaluate(y_true, y_pred):
    y_pred = torch.argmax(y_pred, dim=-1)
    return torch.mean((y_true == y_pred).float()) * 100.


def evaluate_top(y_true, y_pred, top=1):
    y_true = torch.unsqueeze(y_true, dim=-1)
    y_pred = torch.argsort(y_pred, dim=-1, descending=True)[:, 0:top]
    equal = torch.eq(y_true, y_pred)
    acc = torch.mean(torch.any(equal, dim=-1).float())
    return acc * 100.


def one_hot(labels, n_class):
    y = torch.zeros((labels.shape[0], n_class), dtype=torch.int8)
    y[torch.arange(labels.shape[0])[:, None], labels] = 1
    return y


def clustering_metrics(true, pred, name):
    adjusted_rand_score = metrics.adjusted_rand_score(true, pred)

    print("\x1B[4m" + name + "\x1B[0m")
    print('Adjusted Rand Score: {0:.2f}%'.format(adjusted_rand_score * 100))
