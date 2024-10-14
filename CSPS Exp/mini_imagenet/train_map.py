import torch
from setup import *
from network_vtb import Network
from utils import uniform, normalize, evaluate_top
from mini_imagenet_dataset import mini_imagenet_augment_84

train_loader, test_loader = mini_imagenet_augment_84(root=root(), batch_size=64, num_workers=4)
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(device)

network = Network(in_channels=3, classes=100)
network.to(device)
# network.load_state_dict(torch.load('./../weights/mini_imagenet_vtb.h5'))

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
            key = uniform(x_true.shape).to(device)
            x_true = normalize(x_true * key)

            # forward
            y_pred, y_advs = network(x_true, key, unbind_fn=lambda x, y, ch: x * y)
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

torch.save(network.state_dict(), './../weights/mini_imagenet_map.h5')
print('All Done!')
