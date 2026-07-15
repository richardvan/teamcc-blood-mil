#!/usr/bin/env python
"""
gradcam_cnn.py
================
Grad-CAM on a few sample cell images, using the fine-tuned layer4 block of a
saved gen2_cnn checkpoint. Shows which parts of a cell image drove the
model's prediction -- something SVM fundamentally can't provide (it never
sees pixel-level spatial structure, only a pooled feature vector).

Usage:
    python gradcam_cnn.py \
        --checkpoint .../cnn_mil_fold1_max.pt \
        --organized_dir .../organized_data \
        --patient_folder cancer.CBFB_MYH11.AQK \
        --n_images 4 \
        --output_dir ./gradcam_out
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

CLASSES_FALLBACK = ["control", "CBFB_MYH11", "NPM1", "PML_RARA", "RUNX1_RUNX1T1"]

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def gradcam_single_image(model_backbone, classifier, layer4, pooling, img_tensor, device):
    """
    ResNet's backbone.fc was replaced with Identity, so backbone(x) already
    does global average pooling internally and returns a flat (1,2048)
    vector -- there's no spatial feature map left to hook for Grad-CAM at
    that point. Instead we run the conv trunk manually up to layer4 (before
    avgpool) to get the spatial map, then redo avgpool ourselves.
    """
    img_tensor = img_tensor.unsqueeze(0).to(device).requires_grad_(False)
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = layer4.register_forward_hook(fwd_hook)
    h2 = layer4.register_full_backward_hook(bwd_hook)

    # Manually replicate resnet forward up through layer4 (spatial map),
    # then global-average-pool to match what the model was trained on.
    x = model_backbone.conv1(img_tensor)
    x = model_backbone.bn1(x)
    x = model_backbone.relu(x)
    x = model_backbone.maxpool(x)
    x = model_backbone.layer1(x)
    x = model_backbone.layer2(x)
    x = model_backbone.layer3(x)
    x = model_backbone.layer4(x)           # (1, 2048, 7, 7) -- hooked here
    pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (1, 2048), matches backbone.fc=Identity output

    # This single image as its own "bag" -- pooling over a bag of size 1 is a no-op
    # for mean/max/min; min_max would double the vector, so use mean/max checkpoints
    # with Grad-CAM (min_max Grad-CAM isn't well-defined for a single instance).
    bag_vec = pooled
    scores = classifier(bag_vec)
    pred_class = scores.argmax(dim=1).item()

    model_backbone.zero_grad()
    classifier.zero_grad()
    scores[0, pred_class].backward()

    grads = gradients["value"]             # (1, 2048, 7, 7)
    acts = activations["value"]            # (1, 2048, 7, 7)
    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, 2048, 1, 1) -- GAP of gradients
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1, 1, 7, 7)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.detach().squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove()
    h2.remove()
    return cam, pred_class


def unnormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--organized_dir", required=True)
    p.add_argument("--patient_folder", required=True, help="e.g. cancer.CBFB_MYH11.AQK")
    p.add_argument("--n_images", type=int, default=4)
    p.add_argument("--image_ext", default=".tif")
    p.add_argument("--output_dir", default="./gradcam_out")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    pooling = ckpt.get("pooling", "mean")
    if pooling == "min_max":
        print("WARNING: Grad-CAM on a min_max checkpoint treats the image as a "
              "size-1 bag, where min==max, so this still works, but be aware "
              "min_max's 4096-dim classifier weight is being used unchanged.")
    classes = ckpt.get("classes", CLASSES_FALLBACK)

    backbone = models.resnet50(weights=None)
    backbone.fc = torch.nn.Identity()
    in_dim = 4096 if pooling == "min_max" else 2048
    classifier = torch.nn.Linear(in_dim, len(classes))

    # Load matching state dict pieces (state_dict keys are "backbone.xxx" / "classifier.xxx")
    full_sd = ckpt["state_dict"]
    backbone_sd = {k[len("backbone."):]: v for k, v in full_sd.items() if k.startswith("backbone.")}
    classifier_sd = {k[len("classifier."):]: v for k, v in full_sd.items() if k.startswith("classifier.")}
    backbone.load_state_dict(backbone_sd)
    classifier.load_state_dict(classifier_sd)
    backbone.to(device).eval()
    classifier.to(device).eval()
    for p_ in backbone.parameters():
        p_.requires_grad_(True)  # need gradients w.r.t. activations, not weight updates

    paths = sorted(glob.glob(os.path.join(args.organized_dir, args.patient_folder, f"*{args.image_ext}")))[:args.n_images]
    if not paths:
        raise FileNotFoundError(f"No images found in {args.organized_dir}/{args.patient_folder}")

    fig, axes = plt.subplots(2, len(paths), figsize=(4 * len(paths), 8))
    if len(paths) == 1:
        axes = axes.reshape(2, 1)

    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        img_tensor = PREPROCESS(img)
        cam, pred_class = gradcam_single_image(backbone, classifier, backbone.layer4, pooling, img_tensor, device)

        orig = unnormalize(img_tensor)
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f"pred: {classes[pred_class]}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(orig)
        axes[1, i].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, i].set_title("Grad-CAM (layer4)", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle(f"Grad-CAM -- {args.patient_folder} ({pooling} pooling)")
    fig.tight_layout()
    out_path = os.path.join(args.output_dir, f"gradcam_{args.patient_folder}_{pooling}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
