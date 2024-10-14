import torch
import torch.nn as nn

from unet import unet
from pytorch_revgrad import RevGrad


class Network(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        activation = nn.LeakyReLU(0.1)

        self.F_main = nn.Sequential(
            unet.UNet2D(in_channels, in_channels)
        )

        self.F_prediction = nn.Sequential(
            nn.Conv2d(in_channels, 32, (3, 3), padding=(1, 1)), nn.BatchNorm2d(32), activation,
            nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)), nn.BatchNorm2d(64), activation,
            nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)), nn.BatchNorm2d(128), activation,
            nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(2048, 1024), activation, nn.BatchNorm1d(1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512), activation, nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, classes)
        )

        self.F_adversarial = nn.Sequential(
            RevGrad(),
            nn.Conv2d(in_channels, 32, (3, 3), padding=(1, 1)), nn.BatchNorm2d(32), activation,
            nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)), nn.BatchNorm2d(64), activation,
            nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)), nn.BatchNorm2d(128), activation,
            nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(2048, 1024), activation, nn.BatchNorm1d(1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512), activation, nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, classes)
        )

    def forward(self, x, key, ret_data=False):
        x_main = self.F_main(x)

        y_pred = self.F_prediction(x_main / key)

        y_advs = self.F_adversarial(x_main)

        if ret_data:
            return y_pred, y_advs, x_main
        else:
            return y_pred, y_advs


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = Network(in_channels=3, classes=10)
    network.to(device)

    x_ = torch.normal(0, 1, (10, 3, 32, 32), dtype=torch.float32).to(device)
    k_ = torch.normal(0, 1, (10, 3, 32, 32), dtype=torch.float32).to(device)

    out1, out2 = network(x_, k_)

    print(out1.shape)
    print(out2.shape)
