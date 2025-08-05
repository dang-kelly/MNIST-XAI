
# https://captum.ai/
# https://captum.ai/docs/attribution_algorithms
import torch
import matplotlib.pyplot as plt
from captum.attr import (
    Saliency,
    IntegratedGradients,
    InputXGradient,
    GuidedBackprop,
    Deconvolution,
    NoiseTunnel,
    # GuidedGradCam → requires CNN
)
from train import load_and_classify

def visualize_attribution(attr, image_tensor, title="Attribution"):
    attr = attr.squeeze().cpu().detach().numpy()
    image = image_tensor.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(7, 3))

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')

    im = axs[1].imshow(attr, cmap='hot')
    # print(attr)
    axs[1].set_title(title)
    axs[1].axis('off')

    # Add colorbar/legend for attribution values
    cbar = plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label("Attribution Intensity", rotation=270, labelpad=10)

    plt.tight_layout()
    plt.show()

def run_captum(model, image_tensor, method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = image_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

    print(f"Predicted class: {pred_class}")

    if method == "saliency":
        explainer = Saliency(model)
        attr = explainer.attribute(input_tensor, target=pred_class)
        visualize_attribution(attr, image_tensor, "Saliency Map")

        # base = Saliency(model)
        # explainer = NoiseTunnel(base)
        # attr = explainer.attribute(
        #     input_tensor,
        #     nt_type="smoothgrad",
        #     nt_samples=50,
        #     stdevs=0.2,
        #     target=pred_class
        # )
        # visualize_attribution(attr, image_tensor, "SmoothGrad (Saliency)")

    elif method == "ig":
        explainer = IntegratedGradients(model)
        baseline = torch.zeros_like(input_tensor).to(device)
        attr = explainer.attribute(input_tensor, baseline, target=pred_class, n_steps=50)
        visualize_attribution(attr, image_tensor, "Integrated Gradients")

    elif method == "inputxgradient":
        explainer = InputXGradient(model)
        attr = explainer.attribute(input_tensor, target=pred_class)
        visualize_attribution(attr, image_tensor, "Input × Gradient")

    # elif method == "guidedbp":
    #     explainer = GuidedBackprop(model)
    #     attr = explainer.attribute(input_tensor, target=pred_class)
    #     visualize_attribution(attr, image_tensor, "Guided Backpropagation")

    # elif method == "deconv":
    #     explainer = Deconvolution(model)
    #     attr = explainer.attribute(input_tensor, target=pred_class)
    #     visualize_attribution(attr, image_tensor, "Deconvolution")

    else:
        print("Unknown method:", method)

def main():
    model_names=["ffn_mnist.pth","CNN_MNIST.pth"]
    model, image, label = load_and_classify(model_name=model_names[1],index=5)
    print(f"True label: {label}")

    methods = ["saliency", "ig", "inputxgradient", "guidedbp", "deconv"]
    for method in methods:
        print(f"\n Running {method}...")
        run_captum(model, image, method)

if __name__ == "__main__":
    main()
