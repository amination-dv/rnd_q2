import os
import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from rnd.utils.modeling import get_model
from rnd.utils.transform import resize_to_224
from rnd.utils.dataset import NumpyImageFolder
from rnd.utils.augmentation import (
    RandomHorizontalRoll,
    RandomRowSwap,
    RandomApplyNp,
    RandomJitter,
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model, weights = get_model(args.model, args.dense_units, args.dropout)
    model = model.to(device)

    # transform pipeline
    transform = transforms.Compose(
        [
            transforms.Lambda(resize_to_224),
            RandomApplyNp(RandomHorizontalRoll(max_shift=150), p=0.4),
            RandomApplyNp(RandomRowSwap(num_swaps=4), p=0.3),
            # RandomApplyNp(RandomJitter(max_jitter=30), p=0.1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
            transforms.Normalize(
                mean=weights.transforms().mean, std=weights.transforms().std
            ),
        ]
    )

    # Dataset and DataLoader
    full_ds = NumpyImageFolder(root_dir=args.data_dir, transform=transform)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[Epoch {epoch + 1}] Train Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                preds = (outputs > 0.5).long().squeeze()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"[Epoch {epoch + 1}] Validation Accuracy: {acc:.4f}")

    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(
        model.state_dict(), os.path.join(args.save_dir, f"{args.model}_final.pth")
    )
    print(f"✅ Model saved to: {args.save_dir}/{args.model}_final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="resnet", choices=["resnet", "mobilenet"]
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dense-units", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--save-dir", type=str, default="models/saved_models")
    args = parser.parse_args()
    main(args)
