import torch
from setup import *
from math import sqrt
from tqdm import tqdm
from network_vtb import Network
from utils import normalize, evaluate_top
from mini_imagenet_dataset import mini_imagenet_augment_84
from vtb_ops import vtb, normal

train_loader, test_loader = mini_imagenet_augment_84(root=root(), batch_size=25, num_workers=4)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
_, vtb_mod = vtb(batch_size=25, input_dim=84, device=device)
print(device)

network = Network(in_channels=3, classes=100)
network.to(device)
# network.load_state_dict(torch.load('./../weights/mini_imagenet.h5'))

epochs = 100
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader))
loss_function = torch.nn.CrossEntropyLoss()

logger = collections.defaultdict(list)

for epoch in range(1, epochs + 1):
    """ train """
    tic = time.time()
    logger["loss"] = []
    logger["acc@1"] = []
    logger["acc@5"] = []
    network.train()

    for data in tqdm(train_loader):
        x_true, y_true = data[0].to(device), data[1].to(device)
        key = normal(x_true.shape, std=1. / sqrt(84. * 84. * 3.)).to(device)
        x_true = normalize(vtb_mod.binding(x_true, key, ch=3))

        optimizer.zero_grad()

        # forward + loss + backward + optimize
        y_pred, y_advs = network(x_true, key, unbind_fn=vtb_mod.unbinding)
        loss = loss_function(y_pred, y_true) + loss_function(y_advs, y_true)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # evaluate
        acc1 = evaluate_top(y_true, y_pred, top=1)
        acc5 = evaluate_top(y_true, y_pred, top=5)
        logger["loss"].append(loss)
        logger["acc@1"].append(acc1)
        logger["acc@5"].append(acc5)

    logger["train_loss"].append(torch.mean(torch.tensor(logger["loss"])))
    logger["train_acc@1"].append(torch.mean(torch.tensor(logger["acc@1"])))
    logger["train_acc@5"].append(torch.mean(torch.tensor(logger["acc@5"])))

    """ test """

    logger["loss"] = []
    logger["acc@1"] = []
    logger["acc@5"] = []
    network.eval()

    with torch.no_grad():
        for data in test_loader:
            x_true, y_true = data[0].to(device), data[1].to(device)
            key = normal(x_true.shape, std=1. / sqrt(84. * 84. * 3.)).to(device)
            x_true = normalize(vtb_mod.binding(x_true, key, ch=3))

            # forward
            y_pred, y_advs = network(x_true, key, unbind_fn=vtb_mod.unbinding)
            loss = loss_function(y_pred, y_true) + loss_function(y_advs, y_true)

            # evaluate
            acc1 = evaluate_top(y_true, y_pred, top=1)
            acc5 = evaluate_top(y_true, y_pred, top=5)
            logger["loss"].append(loss)
            logger["acc@1"].append(acc1)
            logger["acc@5"].append(acc5)

        logger["test_loss"].append(torch.mean(torch.tensor(logger["loss"])))
        logger["test_acc@1"].append(torch.mean(torch.tensor(logger["acc@1"])))
        logger["test_acc@5"].append(torch.mean(torch.tensor(logger["acc@5"])))

    toc = time.time()

    """ history """

    logger["history"].append(f'Epoch: [{epoch:>3d}/{epochs}], '
                             f'train loss: {logger["train_loss"][-1]:>6.4f}, '
                             f'train acc@1: {logger["train_acc@1"][-1]:>5.2f}%, '
                             f'train acc@5: {logger["train_acc@5"][-1]:>5.2f}%, '
                             f'test loss: {logger["test_loss"][-1]:>6.4f}, '
                             f'test acc@1: {logger["test_acc@1"][-1]:>5.2f}%, '
                             f'test acc@5: {logger["test_acc@5"][-1]:>5.2f}%, '
                             f'etc: {toc - tic:>5.2f}s')
    print(logger["history"][-1])

torch.save(network.state_dict(), './../weights/mini_imagenet_vtb.h5')
print('All Done!')
# Epoch: [100/100], train loss: 6.7920, train acc@1: 42.11%, train acc@5: 69.87%, test loss: 7.8635,
# test acc@1: 45.81%, test acc@5: 73.52%, etc: 4486.82s
