import torch
from setup import *
from math import sqrt
from network_vtb import Network
from dataset import cifar100_augmented
from utils import normalize, evaluate
from vtb_ops import vtb, normal

train_loader, test_loader = cifar100_augmented(root=root(), batch_size=64, num_workers=4)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
_, vtb_mod = vtb(batch_size=64, input_dim=32, device=device)
print(device)

network = Network(in_channels=3, classes=100)
network.to(device)
# network.load_state_dict(torch.load('./../weights/cifar100.h5'))

epochs = 200
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
loss_function = torch.nn.CrossEntropyLoss()

logger = collections.defaultdict(list)

for epoch in range(1, epochs + 1):
    """ train """
    tic = time.time()
    logger["loss"] = []
    logger["acc"] = []
    network.train()

    for data in train_loader:
        x_true, y_true = data[0].to(device), data[1].to(device)
        key = normal(x_true.shape, std=1. / sqrt(32. * 32. * 3.), ).to(device)
        x_true = normalize(vtb_mod.binding(x_true, key, ch=3))

        optimizer.zero_grad()

        # forward + loss + backward + optimize
        y_pred, y_advs = network(x_true, key, unbind_fn=vtb_mod.unbinding)
        loss = loss_function(y_pred, y_true) + loss_function(y_advs, y_true)
        loss.backward()
        optimizer.step()

        # evaluate
        accuracy = evaluate(y_true, y_pred)
        logger["loss"].append(loss)
        logger["acc"].append(accuracy)

    logger["train_loss"].append(torch.mean(torch.tensor(logger["loss"])))
    logger["train_acc"].append(torch.mean(torch.tensor(logger["acc"])))

    """ test """

    logger["loss"] = []
    logger["acc"] = []
    network.eval()

    with torch.no_grad():
        for data in test_loader:
            x_true, y_true = data[0].to(device), data[1].to(device)
            key = normal(x_true.shape, std=1. / sqrt(32. * 32. * 3.), ).to(device)
            x_true = normalize(vtb_mod.binding(x_true, key, ch=3))

            # forward
            y_pred, y_advs = network(x_true, key, unbind_fn=vtb_mod.unbinding)
            loss = loss_function(y_pred, y_true) + loss_function(y_advs, y_true)

            # evaluate
            accuracy = evaluate(y_true, y_pred)
            logger["loss"].append(loss)
            logger["acc"].append(accuracy)

        logger["test_loss"].append(torch.mean(torch.tensor(logger["loss"])))
        logger["test_acc"].append(torch.mean(torch.tensor(logger["acc"])))

    toc = time.time()
    scheduler.step()

    """ history """

    logger["history"].append(f'Epoch: [{epoch:>3d}/{epochs}], '
                             f'train loss: {logger["train_loss"][-1]:>6.4f}, '
                             f'train acc: {logger["train_acc"][-1]:>5.2f}%, '
                             f'test loss: {logger["test_loss"][-1]:>6.4f}, '
                             f'test acc: {logger["test_acc"][-1]:>5.2f}%, '
                             f'etc: {toc - tic:>5.2f}s')
    print(logger["history"][-1])

torch.save(network.state_dict(), './../weights/cifar100_vtb.h5')
print('All Done!')
