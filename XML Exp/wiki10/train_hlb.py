import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import collections
from tqdm import tqdm
from scipy import sparse
from utils import one_hot
from network_sparse import Network
from dataset_sparse import load_dataset, cache_propensity
from metrics import compute_prop_metrics, display_metrics


def main(batch_size, num_workers, epochs, load_weights=False):
    name = "Wiki10-31K"
    train_loader, test_loader, info = load_dataset(train_file=f"../data/{name}/train.txt",
                                                   test_file=f"../data/{name}/test.txt",
                                                   batch_size=batch_size,
                                                   num_workers=num_workers)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Loaded {name} data. Available device: {torch.cuda.device_count()} GPUs")

    network = Network(device, in_features=info["features"], hidden=512, out_features=3000, labels=info["labels"],
                      requires_grad=False, negative=True, drop_rate=0.25)

    if load_weights:
        network.load_state_dict(torch.load(f"./../weights/{name}.h5"))

    network.to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    """ load inverse propensity score """
    inv_prop = cache_propensity(loader=train_loader, info=info, name=name)

    logger = collections.defaultdict(list)

    for epoch in range(1, epochs + 1):
        """ train """
        tic = time.time()
        logger["loss"], logger["acc"] = [], []
        network.train()

        for data in tqdm(train_loader):
            x_true, y_true = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            # forward + loss + backward + optimize
            y_logits = network(x_true)
            loss = network.loss(y_logits, y_true)

            loss.backward()
            optimizer.step()

            logger["loss"].append(loss)

        logger["train_loss"].append(torch.mean(torch.tensor(logger["loss"])))

        toc = time.time()
        train_time = toc - tic

        """ test """
        logger["loss"], logger["metrics"] = [], []
        tic = time.time()

        with torch.no_grad():
            network.eval()
            for data in tqdm(test_loader):
                x_true, y_true = data[0].to(device), data[1].to(device)

                # forward
                y_logits = network(x_true)
                loss = network.loss(y_logits, y_true)

                # evaluate
                y_pred = network.inference(y_logits)
                logger["loss"].append(loss)

                # n_DCG and PSnDCG
                y_true = one_hot(y_true, n_class=info["labels"] + 1)
                y_true = y_true[:, :-1].detach().cpu().numpy()
                y_pred = y_pred[:, :-1].detach().cpu().numpy()

                logger["metrics"].append(compute_prop_metrics(sparse.csr_matrix(y_true),
                                                              sparse.csr_matrix(y_pred),
                                                              inv_prop_scores=inv_prop,
                                                              topk=5))

            logger["test_loss"].append(torch.mean(torch.tensor(logger["loss"])))

        toc = time.time()
        test_time = toc - tic

        logger["history"].append(f"Epoch: [{epoch:>3d}/{epochs}], "
                                 f"train loss: {logger['train_loss'][-1]:>6.4f}, "
                                 f"train etc: {train_time:>5.2f}s, "
                                 f"test loss: {logger['test_loss'][-1]:>6.4f}, "
                                 f"test etc: {test_time:>5.2f}s")

        print(logger["history"][-1])
        display_metrics(logger["metrics"])
        scheduler.step()

    # print("-----Saving Model-----")
    # torch.save(network.state_dict(), f"./../weights/{name}.h5")
    print('All Done!')

    """ 
    ----------Tests with Ordered Retrieval------------
                      1       2       3       4       5
    -----------  ------  ------  ------  ------  ------
    Precision@k  80.203  73.518  66.952  61.205  56.434
    nDCG@k       80.203  75.031  70.056  65.675  61.972
    PSprec@k     10.027  10.304  10.398  10.467  10.530
    PSnDCG@k     10.027  10.237  10.314  10.368  10.415
    All Done!

    Process finished with exit code 0
    """


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    main(batch_size=64, num_workers=4, epochs=25)
