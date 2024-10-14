import torch
from setup import *
from dataset import mnist
from network_vtb import Network
from utils import normalize, evaluate, uniform


train_loader, test_loader = mnist(root=root(), batch_size=64, num_workers=4)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

network = Network(in_channels=1, classes=10)
network.to(device)
# network.load_state_dict(torch.load('./../weights/mnist_map.h5'))

epochs = 50
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
print(len(train_loader))
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader))
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
        key = uniform(x_true.shape).to(device)
        x_true = normalize(x_true * key)

        optimizer.zero_grad()

        # forward + loss + backward + optimize
        y_pred, y_advs = network(x_true, key, unbind_fn=lambda x, y, ch: x * y)
        loss = loss_function(y_pred, y_true) + loss_function(y_advs, y_true)
        loss.backward()
        optimizer.step()
        scheduler.step()

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
            key = uniform(x_true.shape).to(device)
            x_true = normalize(x_true * key)

            # forward
            y_pred, y_advs = network(x_true, key, unbind_fn=lambda x, y, ch: x * y)
            loss = loss_function(y_pred, y_true) + loss_function(y_advs, y_true)

            # evaluate
            accuracy = evaluate(y_true, y_pred)
            logger["loss"].append(loss)
            logger["acc"].append(accuracy)

        logger["test_loss"].append(torch.mean(torch.tensor(logger["loss"])))
        logger["test_acc"].append(torch.mean(torch.tensor(logger["acc"])))

    toc = time.time()

    """ history """

    logger["history"].append(f'Epoch: [{epoch:>3d}/{epochs}], '
                             f'train loss: {logger["train_loss"][-1]:>6.4f}, '
                             f'train acc: {logger["train_acc"][-1]:>5.2f}%, '
                             f'test loss: {logger["test_loss"][-1]:>6.4f}, '
                             f'test acc: {logger["test_acc"][-1]:>5.2f}%, '
                             f'etc: {toc - tic:>5.2f}s')
    print(logger["history"][-1])

torch.save(network.state_dict(), './../weights/mnist_map.h5')
print('All Done!')
