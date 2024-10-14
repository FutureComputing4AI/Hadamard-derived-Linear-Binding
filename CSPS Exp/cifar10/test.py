import torch
import numpy as np
from setup import *
from math import sqrt
from dataset import cifar10_augmented
from utils import normal_mixture, normalize, evaluate, uniform
from vtb_ops import vtb, normal

method = "map"
name = "cifar10_map"
print(method, name)

if method in ["vtb", "map"]:
    from network_vtb import Network
else:
    from network import Network

_, test_loader = cifar10_augmented(root=root(), batch_size=128, num_workers=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_, vtb_mod = vtb(batch_size=128, input_dim=32, device=device)
print(device)

network = Network(in_channels=3, classes=10)
network.to(device)
network.load_state_dict(torch.load(f"./../weights/{name}.h5"))

loss_function = torch.nn.CrossEntropyLoss()
logger = collections.defaultdict(list)

net_in = []
net_out = []
true_cls = []
pred_cls = []

with torch.no_grad():
    logger["loss"] = []
    logger["acc"] = []
    network.eval()
    tic = time.time()

    for data in test_loader:
        x_true, y_true = data[0].to(device), data[1].to(device)
        if method == "vtb":
            key = normal(x_true.shape, std=1. / sqrt(32. * 32. * 3.)).to(device)
            x_true = normalize(vtb_mod.binding(x_true, key, ch=3))
            y_pred, y_advs, x_main = network(x_true, key, unbind_fn=vtb_mod.unbinding, ret_data=True)
        elif method == "map":
            key = uniform(x_true.shape).to(device)
            x_true = normalize(x_true * key)
            y_pred, y_advs, x_main = network(x_true, key, unbind_fn=lambda x, y, ch: x * y, ret_data=True)
        else:
            key = normal_mixture(x_true.shape, std=1. / sqrt(32. * 32. * 3.), ).to(device)
            x_true = normalize(x_true * key)
            y_pred, y_advs, x_main = network(x_true, key, ret_data=True)

        loss = loss_function(y_pred, y_true) + loss_function(y_advs, y_true)

        # store
        net_in.append(x_true.detach().cpu().numpy())
        net_out.append(x_main.detach().cpu().numpy())
        true_cls.append(y_true.detach().cpu().numpy())
        pred_cls.append(torch.argmax(y_pred, dim=-1).detach().cpu().numpy())

        # evaluate
        accuracy = evaluate(y_true, y_pred)
        logger["loss"].append(loss)
        logger["acc"].append(accuracy)

    logger["test_loss"].append(torch.mean(torch.tensor(logger["loss"])))
    logger["test_acc"].append(torch.mean(torch.tensor(logger["acc"])))

toc = time.time()

net_in = np.concatenate(net_in)
net_out = np.concatenate(net_out)
true_cls = np.concatenate(true_cls)
pred_cls = np.concatenate(pred_cls)

print(net_in.shape)
print(net_out.shape)
print(true_cls.shape)
print(pred_cls.shape)

np.save(f"../data_npy/{name}_net_in.npy", net_in)
np.save(f"../data_npy/{name}_net_out.npy", net_out)
np.save(f"../data_npy/{name}_true_cls.npy", true_cls)
np.save(f"../data_npy/{name}_pred_cls.npy", pred_cls)

print(f"test loss: {logger['test_loss'][-1]:>6.4f}, "
      f"test acc: {logger['test_acc'][-1]:>5.2f}%, "
      f"etc: {toc - tic:>5.2f}s")
