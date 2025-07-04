import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from rnd.utils.modeling import get_model
from rnd.utils.transform import resize_to_224
from rnd.utils.dataset import NumpyImageFolder
from rnd.utils.metrics import eval_preds, plot_roc_curve


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, weights = get_model(args.model, args.dense_units, args.dropout)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Lambda(resize_to_224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
            transforms.Normalize(
                mean=weights.transforms().mean, std=weights.transforms().std
            ),
        ]
    )

    test_ds = NumpyImageFolder(root_dir=args.data_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_probs, all_labels, all_paths = [], [], []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = outputs.squeeze().cpu().numpy()
            labels = labels.cpu().numpy()

            if probs.ndim == 0:
                probs = np.expand_dims(probs, 0)
                labels = np.expand_dims(labels, 0)

            all_probs.extend(probs)
            all_labels.extend(labels)

            start = batch_idx * args.batch_size
            end = start + len(labels)
            paths = [test_ds.samples[i][0] for i in range(start, end)]
            all_paths.extend(paths)

    df = pd.DataFrame(
        {"file": all_paths, "probability": all_probs, "true_label": all_labels}
    )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=["resnet", "mobilenet"], default="resnet"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to .pth model file"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to test dataset root"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dense-units", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for classification"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="pred_probs.csv",
        help="Where to save the prediction output",
    )
    args = parser.parse_args()

    df = main(args)
    df.to_csv(args.output_csv, index=False)
    print(f"\n📁 Predictions saved to: {args.output_csv}")

    run_name = Path(args.data_dir).name
    eval_preds(
        df,
        threshold=args.threshold,
        wrong_pred_csv=f"{args.model}_{run_name}_preds.csv",
    )
    y_true = df["true_label"].astype(int).values.flatten()
    y_prob = df["probability"].astype(float).values.flatten()

    best_thresh = plot_roc_curve(y_true, y_prob)
