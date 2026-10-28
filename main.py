"""
=============================================================================
Project 04: Binary Image Classifier (Cats vs Dogs) using PyTorch CNN
=============================================================================
Author     : Gideon Mwiti
Challenge  : 100 PyTorch Projects in 10 Days
Topic      : Convolutional Neural Networks (CNN), ImageFolder, Data Augmentation
Goal       : Classify input images as either Cat or Dog using PyTorch
=============================================================================
"""

import os
import sys
import types
import random

# Ensure safe import of torchvision under all environments
sys.modules['bz2'] = types.SimpleNamespace(open=open)
sys.modules['lzma'] = types.SimpleNamespace(open=open)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

def ensure_dataset_exists(data_dir: str):
    """
    Checks if a Cats vs Dogs image folder dataset exists.
    If not, programmatically generates a clean, structured benchmark
    image dataset in the standard torchvision ImageFolder layout:
      data/train/cats/, data/train/dogs/
      data/val/cats/, data/val/dogs/
    """
    splits = {
        "train": {"cats": 150, "dogs": 150},
        "val": {"cats": 50, "dogs": 50}
    }

    already_exists = all(
        os.path.exists(os.path.join(data_dir, split, cls)) and
        len(os.listdir(os.path.join(data_dir, split, cls))) >= splits[split][cls]
        for split in splits for cls in splits[split]
    )

    if already_exists:
        print("[INFO] Existing Cats vs Dogs ImageFolder dataset found. Reusing cached images.")
        return

    print("[INFO] Synthesizing benchmark Cats vs Dogs dataset into standard ImageFolder layout...")
    random.seed(42)
    np.random.seed(42)

    cat_palettes = [
        (220, 160, 100), (180, 120, 70), (240, 200, 140), (100, 100, 100), (230, 230, 230)
    ]
    dog_palettes = [
        (139, 69, 19), (205, 133, 63), (40, 40, 40), (218, 165, 32), (160, 82, 45)
    ]

    for split in ["train", "val"]:
        for cls in ["cats", "dogs"]:
            target_path = os.path.join(data_dir, split, cls)
            os.makedirs(target_path, exist_ok=True)
            num_samples = splits[split][cls]

            for i in range(num_samples):
                img = Image.new("RGB", (128, 128), color=(
                    random.randint(180, 230),
                    random.randint(200, 240),
                    random.randint(180, 220)
                ))
                draw = ImageDraw.Draw(img)

                if cls == "cats":
                    # Distinct feline geometry: Pointed upright triangular ears, slit pupils, whiskers
                    body_color = random.choice(cat_palettes)
                    # Head
                    draw.ellipse([30, 35, 98, 103], fill=body_color, outline=(40, 40, 40))
                    # Upright pointed ears
                    draw.polygon([(34, 45), (46, 12), (60, 40)], fill=body_color, outline=(40, 40, 40))
                    draw.polygon([(68, 40), (82, 12), (94, 45)], fill=body_color, outline=(40, 40, 40))
                    # Inner ears
                    draw.polygon([(38, 42), (46, 18), (56, 38)], fill=(255, 182, 193))
                    draw.polygon([(72, 38), (82, 18), (90, 42)], fill=(255, 182, 193))
                    # Slit eyes
                    draw.ellipse([46, 56, 56, 70], fill=(50, 205, 50))
                    draw.ellipse([72, 56, 82, 70], fill=(50, 205, 50))
                    draw.line([(51, 57), (51, 69)], fill=(0, 0, 0), width=2)
                    draw.line([(77, 57), (77, 69)], fill=(0, 0, 0), width=2)
                    # Nose and muzzle
                    draw.polygon([(61, 75), (67, 75), (64, 80)], fill=(255, 105, 180))
                    # Whiskers
                    draw.line([(24, 76), (58, 78)], fill=(50, 50, 50), width=1)
                    draw.line([(22, 82), (58, 80)], fill=(50, 50, 50), width=1)
                    draw.line([(70, 78), (104, 76)], fill=(50, 50, 50), width=1)
                    draw.line([(70, 80), (106, 82)], fill=(50, 50, 50), width=1)

                else:
                    # Distinct canine geometry: Floppy/hanging side ears, elongated snout, round pupils
                    body_color = random.choice(dog_palettes)
                    # Head
                    draw.ellipse([28, 30, 100, 102], fill=body_color, outline=(30, 30, 30))
                    # Floppy folded ears hanging down sides
                    draw.ellipse([14, 38, 36, 88], fill=body_color, outline=(30, 30, 30))
                    draw.ellipse([92, 38, 114, 88], fill=body_color, outline=(30, 30, 30))
                    # Round eyes with round pupils
                    draw.ellipse([44, 52, 56, 64], fill=(255, 255, 255))
                    draw.ellipse([72, 52, 84, 64], fill=(255, 255, 255))
                    draw.ellipse([48, 55, 54, 61], fill=(0, 0, 0))
                    draw.ellipse([76, 55, 82, 61], fill=(0, 0, 0))
                    # Prominent dog muzzle & black snout
                    draw.ellipse([50, 68, 78, 92], fill=(230, 210, 180), outline=(50, 50, 50))
                    draw.ellipse([58, 71, 70, 80], fill=(15, 15, 15))
                    # Tongue
                    draw.ellipse([60, 88, 68, 98], fill=(255, 99, 132))

                # Apply slight smoothing for natural imagery
                img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
                img.save(os.path.join(target_path, f"{cls}_{i:03d}.jpg"), quality=95)

    print("[SUCCESS] ImageFolder dataset initialized: 300 Training, 100 Validation images.")

class CatDogCNN(nn.Module):
    """
    4-Stage Deep Convolutional Neural Network (CNN) for Binary Image Classification:
    - Input: 3 x 128 x 128
    - Conv Block 1: 3 -> 32 filters (128 -> 64)
    - Conv Block 2: 32 -> 64 filters (64 -> 32)
    - Conv Block 3: 64 -> 128 filters (32 -> 16)
    - Conv Block 4: 128 -> 128 filters (16 -> 8)
    - Fully Connected: AdaptiveAvgPool -> Linear(128*4*4, 128) -> Dropout(0.5) -> Linear(128, 1)
    """
    def __init__(self, dropout_rate: float = 0.5):
        super(CatDogCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)  # Single logit output for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits

def get_dataloaders(data_dir: str, batch_size: int = 32):
    """
    Loads training and validation sets using torchvision.datasets.ImageFolder
    with data augmentation and ImageNet standard normalization.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.classes, val_dataset

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    """Trains for one epoch, returning average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Evaluates the model on validation data, returning loss, accuracy, and confusion matrix."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_float = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels_float)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            correct += (preds == labels_float).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().squeeze().numpy())
            all_targets.extend(labels.numpy())

    val_loss = running_loss / total
    val_acc = (correct / total) * 100.0

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    all_preds_np = np.array(all_preds).astype(int)
    all_targets_np = np.array(all_targets).astype(int)
    
    tp = ((all_preds_np == 1) & (all_targets_np == 1)).sum()
    tn = ((all_preds_np == 0) & (all_targets_np == 0)).sum()
    fp = ((all_preds_np == 1) & (all_targets_np == 0)).sum()
    fn = ((all_preds_np == 0) & (all_targets_np == 1)).sum()

    cm = np.array([[tn, fp], [fn, tp]])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return val_loss, val_acc, cm, precision, recall, f1

def plot_comprehensive_results(
    history: dict,
    cm: np.ndarray,
    val_dataset,
    model: nn.Module,
    device: torch.device,
    output_path: str
):
    """
    Generates a 4-panel publication visual report:
    1. Sample validation predictions with confidence and visual images
    2. Training vs Validation Binary Cross-Entropy Loss
    3. Training vs Validation Accuracy Progression (%)
    4. Confusion Matrix with Precision, Recall, and F1 annotations
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Project 04: Binary Image Classifier (Cats vs Dogs) - PyTorch CNN Architecture\n"
        f"Validation Accuracy = {history['val_acc'][-1]:.2f}% | Precision = {history['val_prec'][-1]:.4f} | Recall = {history['val_rec'][-1]:.4f} | F1 = {history['val_f1'][-1]:.4f}",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )

    # 1. Sample Predictions (8 images: 4 Cats, 4 Dogs)
    ax1 = axes[0, 0]
    ax1.axis("off")
    ax1.set_title("1. Model Predictions on Validation Images", fontweight="bold", fontsize=11, pad=12)

    sample_subfig = ax1.inset_axes([0.0, 0.0, 1.0, 1.0])
    sample_subfig.axis("off")

    sample_indices = [2, 12, 25, 40, 52, 65, 78, 92]
    class_names = ["Cat", "Dog"]

    for idx_pos, img_idx in enumerate(sample_indices):
        img_tensor, true_label = val_dataset[img_idx]
        with torch.no_grad():
            model.eval()
            output = model(img_tensor.unsqueeze(0).to(device))
            prob = torch.sigmoid(output).item()
            pred_label = 1 if prob >= 0.5 else 0
            conf = (prob if pred_label == 1 else (1.0 - prob)) * 100.0

        row = idx_pos // 4
        col = idx_pos % 4
        sub_ax = sample_subfig.inset_axes([col * 0.25 + 0.01, (1 - row) * 0.48, 0.22, 0.44])

        # De-normalize ImageNet transforms for natural visualization
        inv_img = img_tensor.permute(1, 2, 0).numpy()
        inv_img = inv_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        inv_img = np.clip(inv_img, 0, 1)

        sub_ax.imshow(inv_img)
        sub_ax.axis("off")
        color = "#059669" if pred_label == true_label else "#DC2626"
        sub_ax.set_title(
            f"True: {class_names[true_label]}\nPred: {class_names[pred_label]} ({conf:.1f}%)",
            fontsize=8,
            color=color,
            fontweight="bold"
        )

    # 2. Binary Cross-Entropy Loss Curves
    ax2 = axes[0, 1]
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax2.plot(epochs_range, history["train_loss"], color="#2563EB", linewidth=2.2, label="Train Loss", marker="o")
    ax2.plot(epochs_range, history["val_loss"], color="#EF4444", linewidth=2.2, label="Val Loss", marker="s")
    ax2.set_title("2. Binary Cross-Entropy Loss Convergence", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("BCE Loss")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Accuracy Progression Curves
    ax3 = axes[1, 0]
    ax3.plot(epochs_range, history["train_acc"], color="#2563EB", linewidth=2.2, label="Train Accuracy (%)", marker="o")
    ax3.plot(epochs_range, history["val_acc"], color="#10B981", linewidth=2.2, label="Validation Accuracy (%)", marker="s")
    ax3.set_title("3. Classification Accuracy Progression (%)", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_ylim([45, 105])
    ax3.legend(loc="lower right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Binary Confusion Matrix
    ax4 = axes[1, 1]
    im = ax4.imshow(cm, interpolation="nearest", cmap="Blues")
    ax4.set_title("4. Validation Confusion Matrix", fontweight="bold", fontsize=11)
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    tick_marks = np.arange(2)
    ax4.set_xticks(tick_marks)
    ax4.set_yticks(tick_marks)
    ax4.set_xticklabels(["Cat (Class 0)", "Dog (Class 1)"])
    ax4.set_yticklabels(["Cat (Class 0)", "Dog (Class 1)"])
    ax4.set_xlabel("Predicted Class")
    ax4.set_ylabel("True Ground Truth")

    thresh = cm.max() / 2.0
    labels_cm = [["True Cat (TN)", "False Dog (FP)"], ["False Cat (FN)", "True Dog (TP)"]]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            text = f"{val}\n({labels_cm[i][j]})"
            ax4.text(j, i, text, ha="center", va="center",
                     color="white" if val > thresh else "black",
                     fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visual report saved successfully at: {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    output_plot_path = os.path.join(script_dir, "cat_dog_classification_results.png")

    BATCH_SIZE = 32
    EPOCHS = 8
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=================================================================")
    print("      PROJECT 04: BINARY IMAGE CLASSIFIER (CATS VS DOGS) - CNN    ")
    print("=================================================================")
    print(f"Execution Device    : {DEVICE}")
    print(f"Image Resolution    : 128x128 RGB (3 Channels)")
    print(f"Hyperparameters     : Epochs = {EPOCHS}, Batch Size = {BATCH_SIZE}, LR = {LEARNING_RATE}")
    print("-----------------------------------------------------------------")

    # Step 1: Ensure dataset in standard ImageFolder structure
    ensure_dataset_exists(data_dir)

    # Step 2: Build ImageFolder DataLoaders
    train_loader, val_loader, class_names, val_dataset = get_dataloaders(data_dir=data_dir, batch_size=BATCH_SIZE)
    print(f"Classes Detected    : {class_names} (0 = {class_names[0]}, 1 = {class_names[1]})")

    # Step 3: Initialize CNN Architecture
    model = CatDogCNN(dropout_rate=0.5).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_prec": [],
        "val_rec": [],
        "val_f1": []
    }

    print("-----------------------------------------------------------------")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc (%)':^14} | {'Val Loss':^12} | {'Val Acc (%)':^14} | {'Val F1':^10}")
    print("-----------------------------------------------------------------")

    # Step 4: Training Loop
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, cm, prec, rec, f1 = evaluate(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_prec"].append(prec)
        history["val_rec"].append(rec)
        history["val_f1"].append(f1)

        print(f"{epoch:^7d} | {train_loss:^12.4f} | {train_acc:^14.2f} | {val_loss:^12.4f} | {val_acc:^14.2f} | {f1:^10.4f}")

    print("-----------------------------------------------------------------")
    print(f"Training Complete! Final Validation Accuracy: {val_acc:.2f}% | F1-Score: {f1:.4f}")
    print("=================================================================\n")

    # Step 5: Render and Save Visual Report
    plot_comprehensive_results(
        history=history,
        cm=cm,
        val_dataset=val_dataset,
        model=model,
        device=DEVICE,
        output_path=output_plot_path
    )

if __name__ == "__main__":
    main()
