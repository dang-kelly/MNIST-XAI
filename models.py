
import torch.nn as nn

class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.net = nn.Sequential(
            # flatten because we take in a [1,28,28] vector -> [784] so no mismatch
            nn.Flatten(),               
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)           
        )

    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3),  # 28x28 → 26x26
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=64, kernel_size=5),  # 26x26 → 22x22
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 22x22 → 11x11
            nn.Flatten(),                       # 64 * 11 * 11 = 7744
            nn.Linear(64 * 11 * 11, 128),
            nn.ReLU(),
            nn.Linear(128, 10)                  # final class scores
        )

    def forward(self, x):
        return self.net(x)
    
models_to_train_dict = {"CNN_MNIST":CNN(), "FFNN_MNIST":FFNN()}