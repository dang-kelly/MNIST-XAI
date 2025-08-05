# quantus_xai.py

import torch
from captum.attr import Saliency
from quantus import (
    SensitivityN,
    Infidelity,
    Continuity,
)
from train import FFNN, load_model
from torchvision import datasets, transforms

# Load test image
transform = transforms.ToTensor()
test_set = datasets.MNIST(root="./data", train=False, transform=transform)
image, label = test_set[4]
input_tensor = image.unsqueeze(0)
target_class = label

# Load model
model = load_model()
print(model.device)
print(input_tensor.device)
print(label.device)

# Get attribution
attr_method = Saliency(model)
attr = attr_method.attribute(input_tensor, target=target_class)

# Evaluate with Quantus
scores = {}


# Sensitivity
sensitivity = SensitivityN(nr_samples=10)(model=model, inputs=input_tensor, attributions=attr, targets=target_class)
scores["Sensitivity"] = sensitivity

# Infidelity
infidelity = Infidelity(nr_samples=10, perturb_std=0.02)(model=model, inputs=input_tensor, attributions=attr, targets=target_class)
scores["Infidelity"] = infidelity

# Continuity (example for visual smoothness)
continuity = Continuity()(model=model, inputs=input_tensor, attributions=attr, targets=target_class)
scores["Continuity"] = continuity

# Display results
for metric, value in scores.items():
    print(f"{metric}: {value}")
