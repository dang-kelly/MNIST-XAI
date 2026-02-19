# https://captum.ai/
# https://captum.ai/docs/attribution_algorithms
import numpy as np
import torch
import matplotlib.pyplot as plt
from captum.attr import (
    Saliency,
    IntegratedGradients,
    InputXGradient,
    GuidedBackprop,
    Deconvolution,

    Lime,
    ShapleyValueSampling,
    NeuronConductance,
    LRP,
    # GuidedGradCam → requires CNN 
)
from captum._utils.models.linear_model import SkLearnLinearRegression
from train import load_and_classify


def visualize_attribution(attr, image_tensor, title="Attribution", extra_attrs=None):
    """Original image + main attribution + extra maps for classes 1,2,3,4 (if provided)."""
    import math

    def _to_2d(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.squeeze(x)
        if x.ndim == 3:  # (C,H,W) -> avg channels
            x = x.mean(axis=0)
        return x

    def _norm(a):
        a = a.astype(float)
        mn, mx = a.min(), a.max()
        return np.zeros_like(a) if np.isclose(mn, mx) else (a - mn) / (mx - mn + 1e-8)

    img = _to_2d(image_tensor)
    main = _norm(np.abs(_to_2d(attr)))

    panels = [("Original", img, "gray", False),
              (title,     main, "hot", True)]

    if extra_attrs:
        for cls in (1, 2, 3, 4):
            if cls in extra_attrs:
                panels.append((f"Class {cls}",
                               _norm(np.abs(_to_2d(extra_attrs[cls]))),
                               "hot", True))

    cols = min(3, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.atleast_1d(axs).ravel()

    for i, (t, arr, cmap, cbar_flag) in enumerate(panels):
        im = axes[i].imshow(arr, cmap=cmap)
        axes[i].set_title(t)
        axes[i].axis('off')
        if cbar_flag:
            cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.set_label("Attribution Intensity", rotation=270, labelpad=10)

    for j in range(len(panels), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def run_captum(model, image_tensor, method):
    """
    Compact runner: builds one attribution function for the chosen method,
    computes attr for the predicted class + extras for classes 1,2,3,4.
    """
    # Keep device chosen in load_model(); fall back if needed
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    input_example = image_tensor.unsqueeze(0).to(device)  # [1,1,28,28]
    input_example.requires_grad_(True)

    with torch.no_grad():
        pred_class = int(model(input_example).argmax(dim=1).item())
    print(f"Predicted class: {pred_class}")

    titles = {
        "saliency":        "Saliency Map",
        "ig":              "Integrated Gradients",
        "inputxgradient":  "Input × Gradient",
        "guidedbp":        "Guided Backpropagation",
        "deconv":          "Deconvolution",
    }
    # all methods run almost the same with some exceptions like Integrated gradeints
    def make_attr_fn(meth):
        if meth == "ig":
            expl = IntegratedGradients(model)
            baseline = torch.zeros_like(input_example, device=device)
            return lambda target: expl.attribute(input_example, baseline, target=target, n_steps=50)
        cls_map = {
            "saliency":       Saliency,
            "inputxgradient": InputXGradient,
            "guidedbp":       GuidedBackprop,
            "deconv":         Deconvolution,
        }
        ExplClass = cls_map.get(meth)
        if ExplClass is None:
            raise ValueError(f"Unknown method: {meth}")
        expl = ExplClass(model)
        return lambda target: expl.attribute(input_example, target=target)

    attr_xai_method = make_attr_fn(method)

    # Main attribution (predicted class)
    attr = attr_xai_method(pred_class)

    # Extras (classes 1,2,3,4)
    extra_attrs = {}
    for t in (1, 2, 3, 4):
        try:
            extra_attrs[t] = attr_xai_method(t)
        except Exception as e:
            print(f"[{method}] skipped class {t}: {e}")

    visualize_attribution(attr, image_tensor, titles.get(method, method), extra_attrs)


### more methods
def run_captum2(model, image_tensor, method):
    """
    Compact runner: builds one attribution function for the chosen method,
    computes attr for the predicted class + extras for classes 1,2,3,4.
    """
    # Keep device chosen in load_model(); fall back if needed
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    input_example = image_tensor.unsqueeze(0).to(device)  # [1,1,28,28]
    input_example.requires_grad_(True)

    with torch.no_grad():
        pred_class = int(model(input_example).argmax(dim=1).item())
    print(f"Predicted class: {pred_class}")

    titles = {
        "lime":            "Lime",
        "ig":              "Integrated Gradients",
        "shapley":         "Shapley Values Sampling",
        "nc":              "Neuron Conductance",
        "lrp":             "Layerwise Relevance Propagation",
    }

    # Compute attribution for the chosen method
    def make_attr_fn(meth):
        if meth == "ig":
            expl = IntegratedGradients(model)
            baseline = torch.zeros_like(input_example, device=device)
            return lambda target: expl.attribute(input_example, baseline, target=target, n_steps=50)
        
        elif meth == "shapley":
            expl = ShapleyValueSampling(model)
            return lambda target: expl.attribute(input_example, target=target, n_samples=5) # 5 just for speed, increase for better results
        
        elif meth == "nc":
            layer = model.conv2 if hasattr(model, "conv2") else model.net[1]
            
            # Forward pass through layer
            with torch.no_grad():
                layer_out = layer(input_example)
            layer_out = layer_out[0]

            # Find the most activated neuron for explanation
            flat_index = torch.argmax(layer_out)
            channel, h, w = torch.unravel_index(flat_index, layer_out.shape)
            neuron_selector = (channel.item(), h.item(), w.item())

            print(f"Selected neuron: {neuron_selector}")
            expl = NeuronConductance(model, layer) 
            return lambda target: expl.attribute(input_example, target=target, neuron_selector=neuron_selector)

        elif meth == "lime":
            # Use linear regression as the surrogate model
            expl = Lime(model, interpretable_model=SkLearnLinearRegression())
            
            # Feature mask for LIME: each pixel is a separate feature (784 total)
            f_mask = torch.arange(28 * 28, device=device).reshape(1, 1, 28, 28)  # [1,1,28,28]
            return lambda target: expl.attribute(input_example, target=target, n_samples=2000, feature_mask=f_mask)
        
        elif meth == "lrp":
            expl = LRP(model)
            return lambda target: expl.attribute(input_example, target=target)
        
        else:
            raise ValueError(f"Unknown method: {meth}")

    attr_xai_method = make_attr_fn(method)

    # Main attribution (predicted class)
    attr = attr_xai_method(pred_class)

    # Extras (classes 1,2,3,4)
    extra_attrs = {}
    for t in (1, 2, 3, 4):
        try:
            extra_attrs[t] = attr_xai_method(t)
        except Exception as e:
            print(f"[{method}] skipped class {t}: {e}")

    visualize_attribution(attr, image_tensor, titles.get(method, method), extra_attrs)


def main():
    # Choose which saved model to use
    model_names = ["FFNN_MNIST.pth", "CNN_MNIST.pth"]  # ensure casing matches your saved files
    model, image, label = load_and_classify(model_name=model_names[1], index=5)
    print(f"True label: {label}")

    methods1 = ["saliency", "ig", "inputxgradient", "guidedbp", "deconv"]
    methods2 = ["lime", "shapley", "nc", "lrp"]
    
    for m in methods1 + methods2:
        print(f"\nRunning {m}...")
        if m in methods1:
            run_captum(model, image, m)
        elif m in methods2:
            run_captum2(model, image, m)
        else:
            print(f"Unknown method: {m}")


if __name__ == "__main__":
    main()
