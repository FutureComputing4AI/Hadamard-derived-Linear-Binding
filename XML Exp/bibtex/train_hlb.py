import time
import torch
import collections
from scipy import sparse
from utils import evaluate
from network import Network
from dataset_fast import load_dataset
from metrics import compute_inv_propensity, compute_prop_metrics, display_metrics

name = "bibtex"
torch.random.manual_seed(0)
train_loader, test_loader, info = load_dataset(data_file=f"../data/{name.capitalize()}/{name.capitalize()}_data.txt",
                                               train_file=f"../data/{name.capitalize()}/{name}_trSplit.txt",
                                               test_file=f"../data/{name.capitalize()}/{name}_tstSplit.txt",
                                               batch_size=64,
                                               num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Loaded {name} data. Available device: {device}.")

network = Network(in_features=info["features"], hidden=512, out_features=400, labels=info["labels"], scale=0.25)
network.to(device)
# network.load_state_dict(torch.load(f"./../weights/{name}.h5"))

epochs = 25
loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
logger = collections.defaultdict(list)

for epoch in range(1, epochs + 1):
    """ train """
    tic = time.time()
    logger["loss"], logger["acc"] = [], []
    network.train()

    for data in train_loader:
        x_true, y_true = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        # forward + loss + backward + optimize
        y_logits = network(x_true)
        loss = network.loss(y_logits, y_true)

        loss.backward()
        optimizer.step()

        # evaluate
        y_pred = network.inference(y_logits)
        accuracy = evaluate(y_true, y_pred)
        logger["loss"].append(loss)
        logger["acc"].append(accuracy)

    logger["train_loss"].append(torch.mean(torch.tensor(logger["loss"])))
    logger["train_acc"].append(torch.mean(torch.tensor(logger["acc"])))

    """ history """
    toc = time.time()
    logger["history"].append(f'Epoch: [{epoch:>3d}/{epochs}], '
                             f'train loss: {logger["train_loss"][-1]:>6.4f}, '
                             f'train acc: {logger["train_acc"][-1]:>5.2f}%, '
                             f'etc: {toc - tic:>5.2f}s')
    print(logger["history"][-1])
    scheduler.step()

# print("-----Saving Model-----")
# torch.save(network.state_dict(), f"./../weights/{name}.h5")

""" inverse propensity score """
train_labels = torch.concatenate([y for _, y in train_loader]).numpy()
inv_prop = compute_inv_propensity(train_labels, A=0.55, B=1.5)

""" test """
logger["loss"], logger["metrics"] = [], []
tic = time.time()

with torch.no_grad():
    network.eval()
    for data in test_loader:
        x_true, y_true = data[0].to(device), data[1].to(device)

        # forward
        y_logits = network(x_true)
        loss = network.loss(y_logits, y_true)
        y_pred = network.inference(y_logits)

        # evaluate
        logger["loss"].append(loss)

        # n_DCG and PSnDCG
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        logger["metrics"].append(compute_prop_metrics(sparse.csr_matrix(y_true),
                                                      sparse.csr_matrix(y_pred),
                                                      inv_prop_scores=inv_prop,
                                                      topk=5))

    logger["test_loss"].append(torch.mean(torch.tensor(logger["loss"])))

toc = time.time()

print(f'test loss: {logger["test_loss"][-1]:>6.4f}, '
      f'etc: {toc - tic:>4.2f}s')

display_metrics(logger["metrics"])
print('All Done!')
