"""
Train CEFR scoring model on Modal.

Usage:
    uv run modal run train.py              # Full training
    uv run modal run train.py --test-run   # Quick test (5 min)
"""
import json

import modal

# ============================================================
# Modal Configuration
# ============================================================

app = modal.App("cefr-workshop")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=5.14.1",
        "scikit-learn>=1.3.0",
        "sentencepiece>=0.1.99",  # Required for DeBERTa tokenizer
    )
    # Add our code
    .add_local_file("model.py", "/app/model.py")
    .add_local_dir("data", "/app/data")
)

# Persistent storage for trained models
volume = modal.Volume.from_name("cefr-models", create_if_missing=True)


# ============================================================
# Training Function
# ============================================================

@app.function(
    image=image,
    gpu="A10G",  # NVIDIA A10G: 24GB VRAM, good price/performance
    timeout=3600,  # 1 hour max
    volumes={"/vol": volume},
)
def train(
    test_run: bool = False,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    num_epochs: int = 10,
    max_length: int = 512,
):
    """
    Train the CEFR model.

    Args:
        test_run: If True, train for 1 epoch on small subset
        learning_rate: AdamW learning rate (2e-5 is standard for fine-tuning)
        batch_size: Samples per gradient update
        num_epochs: Full passes through training data
        max_length: Max tokens (512 captures most essays)
    """
    import random

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from sklearn.metrics import mean_absolute_error

    # Import our model
    import sys
    sys.path.insert(0, "/app")
    from model import CEFRModel

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 60)
    print("CEFR Model Training")
    print("=" * 60)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Test run: {test_run}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs if not test_run else 1}")
    print(f"Seed: {SEED}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------

    def load_jsonl(path: str) -> list[dict]:
        from pathlib import Path
        if not Path(path).exists():
            raise FileNotFoundError(
                f"\n❌ Data file not found: {path}\n\n"
                "To fix this, run:\n"
                "  uv run python prepare_data.py --input-dir /path/to/corpus/whole-corpus\n\n"
                "See README.md for details on obtaining the W&I corpus."
            )
        with open(path) as f:
            return [json.loads(line) for line in f]

    train_data = load_jsonl("/app/data/train.jsonl")
    dev_data = load_jsonl("/app/data/dev.jsonl")

    if test_run:
        train_data = train_data[:100]
        dev_data = dev_data[:50]

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(dev_data)}")

    # --------------------------------------------------------
    # Tokenization (pre-tokenize once for efficiency)
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    class CEFRDataset(Dataset):
        """Pre-tokenizes all samples once to avoid re-tokenizing each epoch."""

        def __init__(self, data: list[dict]):
            print(f"  Tokenizing {len(data)} samples...")
            self.encodings = tokenizer(
                [item["input"] for item in data],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            self.labels = torch.tensor(
                [item["target"] for item in data], dtype=torch.float32
            )

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    train_loader = DataLoader(
        CEFRDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
    )
    dev_loader = DataLoader(
        CEFRDataset(dev_data),
        batch_size=batch_size,
    )

    # --------------------------------------------------------
    # Model Setup
    # --------------------------------------------------------

    # .float() ensures all params are FP32 — DeBERTa-v3 stores some internal
    # weights in FP16, which can cause NaN during forward passes without this.
    model = CEFRModel().float().to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer: AdamW with weight decay (regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # Learning rate scheduler: linear warmup then decay
    num_training_steps = len(train_loader) * (1 if test_run else num_epochs)
    num_warmup_steps = num_training_steps // 10

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Loss function: Mean Squared Error
    loss_fn = torch.nn.MSELoss()

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------

    best_dev_mae = float("inf")
    epochs = 1 if test_run else num_epochs

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Move data to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            predictions = model(input_ids, attention_mask)
            loss = loss_fn(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                predictions = model(input_ids, attention_mask)

                all_preds.extend(predictions.cpu().tolist())
                all_labels.extend(labels.tolist())

        dev_mae = mean_absolute_error(all_labels, all_preds)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Dev MAE: {dev_mae:.4f}")

        # Save best model
        if dev_mae < best_dev_mae:
            best_dev_mae = dev_mae
            torch.save(model.state_dict(), "/vol/best_model.pt")
            tokenizer.save_pretrained("/vol/tokenizer")
            print("  ✅ New best model saved!")

    # --------------------------------------------------------
    # Final Evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Dev MAE: {best_dev_mae:.4f}")
    print("Model saved to /vol/best_model.pt")
    print("=" * 60)

    # Commit volume to persist
    volume.commit()

    return {"best_dev_mae": best_dev_mae}


# ============================================================
# Entry Point
# ============================================================

@app.local_entrypoint()
def main(test_run: bool = False):
    """Run training."""
    result = train.remote(test_run=test_run)
    print(f"Result: {result}")
