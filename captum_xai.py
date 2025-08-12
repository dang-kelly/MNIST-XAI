
# https://captum.ai/
# https://captum.ai/docs/attribution_algorithms
import torch
import matplotlib.pyplot as plt
import math
import numpy as np
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


def visualize_attribution(attr, image_tensor, title="Attribution", extra_attrs=None):
    """
    Shows: Original image, main attribution, and extra attributions for classes 1,2,3,4
    if provided via extra_attrs={class_id: tensor}. Normalizes each map for display.
    """

    def _to_2d(x):
        # Accepts torch.Tensor or np.ndarray with shapes:
        # (1,C,H,W) or (C,H,W) or (H,W). Returns (H,W) np.ndarray.
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.squeeze(x)
        if x.ndim == 3:           # (C,H,W) -> avg channels
            x = x.mean(axis=0)
        return x

    def _norm(arr):
        arr = arr.astype(float)
        mn, mx = arr.min(), arr.max()
        return np.zeros_like(arr) if np.isclose(mn, mx) else (arr - mn) / (mx - mn + 1e-8)

    main_attr = _norm(np.abs(_to_2d(attr)))
    img       = _to_2d(image_tensor)

    # Panels: (title, array, cmap, colorbar?)
    panels = [("Original", img, "gray", False),
              (title,     main_attr, "hot", True)]

    if extra_attrs:
        for cls_id in [1, 2, 3, 4]:
            if cls_id in extra_attrs:
                cls_attr = _norm(np.abs(_to_2d(extra_attrs[cls_id])))
                panels.append((f"Class {cls_id}", cls_attr, "hot", True))

    n = len(panels)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))

    # Normalize axs indexing
    if rows == 1 and cols == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = np.array([axs])
    elif cols == 1:
        axs = np.array([[ax] for ax in axs])

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axs[r, c]
            if idx < n:
                t, arr, cmap, cbar_flag = panels[idx]
                im = ax.imshow(arr, cmap=cmap)
                ax.set_title(t)
                ax.axis('off')
                if cbar_flag:
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Attribution Intensity", rotation=270, labelpad=10)
            else:
                ax.axis('off')
            idx += 1

    plt.tight_layout()
    plt.show()

def run_captum(model, image_tensor, method):
    """
    Computes attribution for the predicted class, and ALSO for classes 1,2,3,4
    (shown in visualize_attribution). Keeps your original method selector.
    """
    # Keep the model on whatever device load_model() chose
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()  # ensure eval for stable attributions

    # image_tensor is [1,28,28] from load_and_classify
    input_tensor = image_tensor.unsqueeze(0).to(model_device)  # [1,1,28,28]
    input_tensor.requires_grad_()

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = int(output.argmax(dim=1).item())

    print(f"Predicted class: {pred_class}")

    extra_targets = [1, 2, 3, 4]
    extra_attrs = {}

    if method == "saliency":
        explainer = Saliency(model)
        attr = explainer.attribute(input_tensor, target=pred_class)
        for t in extra_targets:
            try:
                extra_attrs[t] = explainer.attribute(input_tensor, target=t)
            except Exception as e:
                print(f"[saliency] skipped class {t}: {e}")
        visualize_attribution(attr, image_tensor, "Saliency Map", extra_attrs)

    elif method == "ig":
        explainer = IntegratedGradients(model)
        baseline = torch.zeros_like(input_tensor, device=model_device)
        attr = explainer.attribute(input_tensor, baseline, target=pred_class, n_steps=50)
        for t in extra_targets:
            try:
                extra_attrs[t] = explainer.attribute(input_tensor, baseline, target=t, n_steps=50)
            except Exception as e:
                print(f"[ig] skipped class {t}: {e}")
        visualize_attribution(attr, image_tensor, "Integrated Gradients", extra_attrs)

    elif method == "inputxgradient":
        explainer = InputXGradient(model)
        attr = explainer.attribute(input_tensor, target=pred_class)
        for t in extra_targets:
            try:
                extra_attrs[t] = explainer.attribute(input_tensor, target=t)
            except Exception as e:
                print(f"[inputxgradient] skipped class {t}: {e}")
        visualize_attribution(attr, image_tensor, "Input × Gradient", extra_attrs)

    elif method == "guidedbp":
        explainer = GuidedBackprop(model)
        attr = explainer.attribute(input_tensor, target=pred_class)
        for t in extra_targets:
            try:
                extra_attrs[t] = explainer.attribute(input_tensor, target=t)
            except Exception as e:
                print(f"[guidedbp] skipped class {t}: {e}")
        visualize_attribution(attr, image_tensor, "Guided Backpropagation", extra_attrs)

    elif method == "deconv":
        explainer = Deconvolution(model)
        attr = explainer.attribute(input_tensor, target=pred_class)
        for t in extra_targets:
            try:
                extra_attrs[t] = explainer.attribute(input_tensor, target=t)
            except Exception as e:
                print(f"[deconv] skipped class {t}: {e}")
        visualize_attribution(attr, image_tensor, "Deconvolution", extra_attrs)

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
