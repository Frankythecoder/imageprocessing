"""Train on HAM10000 metadata and image folders without packaging the dataset."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model import CLASSES, SkinLesionCNN, TRAIN_TRANSFORM, VALID_TRANSFORM


class HamDataset(Dataset):
    def __init__(self, records: pd.DataFrame, image_paths: dict[str, Path], transform):
        self.records = records.reset_index(drop=True)
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        with Image.open(self.image_paths[row.image_id]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASSES.index(row.dx)


def load_records(data_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    metadata = data_dir / "HAM10000_metadata.csv"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata}")
    records = pd.read_csv(metadata)
    required = {"image_id", "lesion_id", "dx"}
    if not required.issubset(records.columns):
        raise ValueError(f"Metadata must contain: {', '.join(sorted(required))}")
    if records.empty or records.image_id.isna().any() or records.lesion_id.isna().any():
        raise ValueError("Metadata must contain nonempty image and lesion identifiers")
    if not set(records.dx).issubset(CLASSES):
        raise ValueError("Metadata contains unrecognized diagnosis labels")
    image_paths = {}
    for path in data_dir.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_paths[path.stem] = path
    missing = set(records.image_id) - set(image_paths)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; example: {next(iter(missing))}")
    return records, image_paths


def split_by_lesion(records: pd.DataFrame, validation_fraction: float, seed: int):
    """Keep all images of a lesion in one split to prevent validation leakage."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    lesions = records[["lesion_id", "dx"]].drop_duplicates("lesion_id")
    validation_lesions = set()
    for _, group in lesions.groupby("dx"):
        shuffled = group.sample(frac=1, random_state=seed)
        count = max(1, round(len(group) * validation_fraction)) if len(group) > 1 else 0
        validation_lesions.update(shuffled.head(min(count, len(group) - 1)).lesion_id)
    validation_mask = records.lesion_id.isin(validation_lesions)
    train_records = records[~validation_mask].copy()
    validation_records = records[validation_mask].copy()
    if validation_records.empty:
        raise ValueError("Not enough distinct lesions for a validation split")
    return train_records, validation_records


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    records, image_paths = load_records(args.data_dir)
    train_records, validation_records = split_by_lesion(records, args.validation_fraction, args.seed)
    train_dataset = HamDataset(train_records, image_paths, TRAIN_TRANSFORM)
    validation_dataset = HamDataset(validation_records, image_paths, VALID_TRANSFORM)
    counts = train_records.dx.value_counts()
    weights = torch.tensor([1.0 / counts[label] for label in train_records.dx], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkinLesionCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Training: {len(train_dataset)} | validation: {len(validation_dataset)} | device: {device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for images, labels in validation_loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                correct += (predictions == labels).sum().item()
                total += len(labels)
        accuracy = correct / total
        print(f"Epoch {epoch}/{args.epochs}: loss={total_loss / len(train_dataset):.4f}, val_accuracy={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({"model_state": model.state_dict(), "classes": list(CLASSES)}, args.output)
            print(f"Saved best checkpoint to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the HAM10000 seven-class CNN")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    train(args)


if __name__ == "__main__":
    main()
