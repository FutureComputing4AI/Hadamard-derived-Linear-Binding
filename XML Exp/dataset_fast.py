import torch
from torch.utils.data import Dataset, DataLoader


class XMLDataset(Dataset):
    def __init__(self, features, labels, index_list, num_features, num_labels):
        self.num_features = num_features
        self.num_labels = num_labels
        index_list = torch.tensor(index_list, dtype=torch.int32)

        self.features = features[index_list]
        self.labels = labels[index_list]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def load_dataset(data_file, train_file, test_file, batch_size=64, cross_valid=0, num_workers=0):
    with open(data_file, mode="r") as f:
        data_lines = f.readlines()
        header = data_lines[0].split(" ")
        data = data_lines[1:]

    with open(train_file, mode="r") as f:
        train_lines = f.readlines()

    with open(test_file, mode="r") as f:
        test_lines = f.readlines()

    data_size, num_features, num_labels = int(header[0]), int(header[1]), int(header[2])
    print(f"Dataset size: {data_size}, " +
          f"No. of features: {num_features}, " +
          f"No. of labels: {num_labels}")

    labels = torch.zeros((data_size, num_labels), dtype=torch.int32)
    features = torch.zeros((data_size, num_features), dtype=torch.float32)

    for i, d in enumerate(data):
        d = d.split(" ")
        try:
            for idx in d[0].split(","):
                labels[i, int(idx)] = 1

            for feats in d[1:]:
                idx, val = feats.split(":")
                features[i, int(idx)] = float(val)
        except ValueError:
            labels[i, :] = labels[i - 1, :]
            features[i, :] = features[i - 1, :]

    train_index = [int(line.split(" ")[cross_valid]) - 1 for line in train_lines]
    test_index = [int(line.split(" ")[cross_valid]) - 1 for line in test_lines]

    train_set = XMLDataset(features, labels, train_index, num_features, num_labels)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = XMLDataset(features, labels, test_index, num_features, num_labels)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    info = {"features": num_features, "labels": num_labels}

    return train_loader, test_loader, info


if __name__ == '__main__':
    train, test, info_ = load_dataset(data_file="data/Bibtex/Bibtex_data.txt",
                                      train_file="data/Bibtex/bibtex_trSplit.txt",
                                      test_file="data/Bibtex/bibtex_tstSplit.txt",
                                      batch_size=64,
                                      cross_valid=0,
                                      num_workers=0)

    for x, y in train:
        print(x.shape)
        print(y.shape)
        print(x)
        print(y)
        break

    for x, y in test:
        print(x.shape)
        print(y.shape)
        print(x)
        print(y)
        break
