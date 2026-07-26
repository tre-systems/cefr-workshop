"""
Evaluate trained model on test set.

Usage:
    uv run modal run evaluate.py
"""
import json
import modal

app = modal.App("cefr-eval")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=5.14.1",
        "scikit-learn>=1.3.0",
        "sentencepiece>=0.1.99",
    )
    .add_local_file("model.py", "/app/model.py")
    .add_local_dir("data", "/app/data")
)

volume = modal.Volume.from_name("cefr-models")


@app.function(
    image=image,
    gpu="T4",  # Smaller GPU is fine for inference
    volumes={"/vol": volume},
)
def evaluate():
    """Evaluate model on held-out test set."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer
    from sklearn.metrics import mean_absolute_error, cohen_kappa_score

    import sys
    sys.path.insert(0, "/app")
    from model import CEFRModel, score_to_cefr, CEFR_TO_SCORE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model...")
    model = CEFRModel().float()
    model.load_state_dict(
        torch.load("/vol/best_model.pt", map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("/vol/tokenizer")

    # Load test data
    with open("/app/data/test.jsonl") as f:
        test_data = [json.loads(line) for line in f]

    print(f"Evaluating on {len(test_data)} test samples...")

    # Pre-tokenize and batch for efficient GPU inference
    class TestDataset(Dataset):
        def __init__(self, data):
            self.encodings = tokenizer(
                [item["input"] for item in data],
                truncation=True,
                padding="max_length",
                max_length=512,
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

    test_loader = DataLoader(TestDataset(test_data), batch_size=32)

    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            preds = model(input_ids, attention_mask)
            # Clamp to valid CEFR range (matches production behavior in serve.py)
            preds = preds.clamp(1.0, 6.0)

            predictions.extend(preds.cpu().tolist())
            actuals.extend(batch["labels"].tolist())

    print(f"  Processed {len(predictions)} samples")

    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)

    # Convert to CEFR levels for QWK
    pred_cefr = [score_to_cefr(p) for p in predictions]
    actual_cefr = [score_to_cefr(a) for a in actuals]

    # Map CEFR to integers for kappa (derived from CEFR_TO_SCORE)
    cefr_to_int = {level: int(score) for level, score in CEFR_TO_SCORE.items()}
    pred_int = [cefr_to_int[c] for c in pred_cefr]
    actual_int = [cefr_to_int[c] for c in actual_cefr]

    qwk = cohen_kappa_score(actual_int, pred_int, weights="quadratic")

    # Accuracy metrics
    exact_match = sum(p == a for p, a in zip(pred_cefr, actual_cefr)) / len(actuals)
    adjacent = sum(
        abs(cefr_to_int[p] - cefr_to_int[a]) <= 1
        for p, a in zip(pred_cefr, actual_cefr)
    ) / len(actuals)

    # Per-level analysis
    from collections import defaultdict
    level_errors = defaultdict(list)
    for pred, actual, actual_cefr_label in zip(predictions, actuals, actual_cefr):
        level_errors[actual_cefr_label].append(abs(pred - actual))

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples:           {len(test_data)}")
    print(f"MAE:               {mae:.3f}")
    print(f"QWK:               {qwk:.3f}")
    print(f"Exact Accuracy:    {exact_match:.1%}")
    print(f"Adjacent Accuracy: {adjacent:.1%}")
    print("\nPer-Level MAE:")
    for cefr in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        if level_errors[cefr]:
            level_mae = sum(level_errors[cefr]) / len(level_errors[cefr])
            print(f"  {cefr}: {level_mae:.3f} (n={len(level_errors[cefr])})")
    print("=" * 60)

    return {
        "mae": mae,
        "qwk": qwk,
        "exact_accuracy": exact_match,
        "adjacent_accuracy": adjacent,
    }


@app.local_entrypoint()
def main():
    result = evaluate.remote()
    print(f"\nFinal Result: {result}")
