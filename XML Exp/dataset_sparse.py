import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from metrics import compute_inv_propensity


class XMLDataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        with open(data_file, mode="r") as f:
            rows = f.readlines()
            head = rows[0].split(" ")
            data = rows[1:]

        data_size, num_features, num_labels = int(head[0]), int(head[1]), int(head[2])

        """ cleanup data """
        max_labels = 0
        max_features = 0
        for d in data:
            if d.split(" ")[0] == "":
                data.remove(d)
                continue
            t = d.split(" ")
            max_labels = max(max_labels, len(t[0].split(",")))
            max_features = max(max_features, len(t[1:]))

        self.data = data
        self.data_size = len(data)
        self.num_features = num_features
        self.num_labels = num_labels
        self.max_labels = max_labels
        self.max_features = max_features
        print(f"Max features: {self.max_features}")

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        features = torch.zeros((self.num_features,), dtype=torch.float32)
        labels = torch.ones((self.max_labels,), dtype=torch.int32) * self.num_labels

        data = self.data[index].split(" ")

        for n, idx in enumerate(data[0].split(",")):
            labels[n] = int(idx)

        for fts in data[1:]:
            idx, val = fts.split(":")
            features[int(idx)] = float(val)

        return features, labels
        # return torch.tensor(features, dtype=torch.float16), labels


def cache_propensity(loader, info, name, root=".."):
    print(f"Dataset size: {info['data_size']}, " +
          f"No. of features: {info['features']}, " +
          f"No. of labels: {info['labels']}")

    if os.path.exists(f"{root}/cache/{name}_inv_prop.npy"):
        print(f"~cached propensity score for {name} already exists!")
        return np.load(f"{root}/cache/{name}_inv_prop.npy")

    data_size = info["data_size"]
    num_labels = info["labels"]
    train_labels = np.zeros((data_size, num_labels + 1), dtype=np.int8)

    labels = [y.numpy() for _, y in tqdm(loader)]
    labels = np.concatenate(labels, axis=0)

    for n in range(data_size):
        train_labels[n][labels[n]] = 1
    train_labels = train_labels[:, :-1]

    if name == "wikipedia":
        A, B = 0.50, 0.40
    elif name == "amazon670k":
        A, B = 0.60, 2.60
    else:
        A, B = 0.55, 1.50

    print(f"Setting A={A} and B={B}")
    inv_prop = compute_inv_propensity(train_labels, A=A, B=B)
    inv_prop = np.asarray(inv_prop, dtype=np.float32)

    print(f"Train labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")
    print(f"Inv propensity score shape: {inv_prop.shape}, dtype: {inv_prop.dtype}")
    print(f"~cached propensity score for {name} dataset of size: {sys.getsizeof(inv_prop) / (1024 ** 2):.2f} GB.")
    np.save(f"{root}/cache/{name}_inv_prop.npy", inv_prop)
    return inv_prop


def load_dataset(train_file, test_file, batch_size, num_workers=0, pin_memory=False):
    train_set = XMLDataset(train_file)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    test_set = XMLDataset(test_file)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)

    info = {"data_size": train_set.data_size, "features": train_set.num_features, "labels": train_set.num_labels}

    return train_loader, test_loader, info


if __name__ == '__main__':
    train_, test, io = load_dataset(train_file=f"./data/Amazon670K.bow/train.txt",
                                    test_file=f"./data/Amazon670K.bow/test.txt",
                                    batch_size=64,
                                    num_workers=0)

    cache_propensity(train_, io, "amazon670k1", root='.')
