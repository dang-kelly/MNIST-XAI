# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import models_to_train_dict
from pathlib import Path

# 1. Define the FFNN model
# class FFNN(nn.Module):
#     def __init__(self):
#         super(FFNN, self).__init__()
#         self.net = nn.Sequential(
#             # flatten because we take in a [1,28,28] vector -> [784] so no mismatch
#             nn.Flatten(),               
#             nn.Linear(784, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 10)           
#         )

#     def forward(self, x):
#         return self.net(x)

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3),  # 28x28 → 26x26
#             nn.ReLU(),
#             nn.Conv2d(in_channels=20, out_channels=64, kernel_size=5),  # 26x26 → 22x22
#             nn.ReLU(),
#             nn.MaxPool2d(2),                    # 22x22 → 11x11
#             nn.Flatten(),                       # 64 * 11 * 11 = 7744
#             nn.Linear(64 * 11 * 11, 128),
#             nn.ReLU(),
#             nn.Linear(128, 10)                  # final class scores
#         )

#     def forward(self, x):
#         return self.net(x)
# 2. Load MNIST dataset
def load_MNIST(batch_size=64):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader


def train_the_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            predicted = pred.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total

def load_model(model_path="FFNN_MNIST.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: THIS WONT LOAD THE RIGHT MODEL DEPENDING ON MODEL PATH, FIX!!!
    model_name = Path(model_path).stem
    model = models_to_train_dict[model_name].to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def classify_and_show(model, image_tensor, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if image_tensor.ndim != 3 or image_tensor.shape[1:] != (28, 28):
        raise ValueError("Expected shape [1, 28, 28], got {}".format(image_tensor.shape))

    input_tensor = image_tensor.unsqueeze(0).to(device)  # Shape: [1, 1, 28, 28]

    with torch.no_grad():
        output = model(input_tensor)
        predicted_label = output.argmax(dim=1).item()

    # Display the image
    plt.imshow(image_tensor.squeeze().cpu(), cmap='gray')
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

    print(f"Predicted Label: {predicted_label}")
    return predicted_label

def load_and_classify(index = 0, model_name = "FFNN_MNIST.pth"):
    transform = transforms.ToTensor()
    test_set = datasets.MNIST(root="./data", train=False, transform=transform)
    image, label = test_set[index]  

    # Load model and classify
    model = load_model(model_name)
    classify_and_show(model, image)
    return model, image, label

def train_and_save(model_path = "FFNN_MNIST.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device type: {device}")
    model_name = Path(model_path).stem
    model = models_to_train_dict[model_name].to(device)
    train_loader, test_loader = load_MNIST()
    # Adam is chosen over vanilla SGD due to its adaptive learning rate and momentum
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        train_loss = train_the_model(model, train_loader, criterion, optimizer, device)
        acc = test(model, test_loader, device)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Accuracy = {acc:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def main():
    train_and_save(model_path="CNN_MNIST.pth")
    load_and_classify(model_name="CNN_MNIST.pth")

if __name__ == "__main__":
    main()
