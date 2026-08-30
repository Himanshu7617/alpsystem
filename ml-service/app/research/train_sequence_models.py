"""Neural Sequence Models (LSTM & Transformer) for ALP Interaction Sequences."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from app.model.features import BASE_FEATURES

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class SequenceDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
            self.labels = torch.tensor(labels, dtype=torch.float32)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]

    def pad_collate(batch):
        (seqs, labels) = zip(*batch)
        lens = [len(s) for s in seqs]
        padded_seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
        return padded_seqs, torch.tensor(labels, dtype=torch.float32), torch.tensor(lens)

    class InteractionLSTM(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x, lengths):
            packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, (hn, _) = self.lstm(packed)
            logits = self.fc(hn[-1])
            return torch.sigmoid(logits).squeeze(-1)

    class SequenceTransformer(nn.Module):
        def __init__(self, input_dim: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
            super().__init__()
            self.proj = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True, dropout=0.1)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x, lengths):
            B, S, _ = x.shape
            mask = torch.arange(S, device=x.device)[None, :] >= lengths[:, None]
            h = self.proj(x)
            out = self.transformer(h, src_key_padding_mask=mask)
            # Pool using last valid sequence position
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
            last_out = out.gather(1, idx).squeeze(1)
            logits = self.fc(last_out)
            return torch.sigmoid(logits).squeeze(-1)


def build_sequential_data(df: pd.DataFrame):
    """Converts event-level dataframe into sequence history arrays per target event."""
    df = df.sort_values(["learner_id", "event_index"]).copy()
    df["next_correct"] = df.groupby("learner_id")["correct"].shift(-1)
    valid_df = df.dropna(subset=["next_correct"])

    sequences = []
    labels = []
    learner_ids = []

    for _, row in valid_df.iterrows():
        l_id = row["learner_id"]
        e_idx = row["event_index"]
        # Pull history up to current event for this learner
        history = df[(df["learner_id"] == l_id) & (df["event_index"] <= e_idx)][BASE_FEATURES].astype(float).values
        sequences.append(history)
        labels.append(int(row["next_correct"]))
        learner_ids.append(l_id)

    return sequences, labels, np.array(learner_ids)


def train_eval_neural_model(model_cls, sequences, labels, train_indices, test_indices, input_dim, seed=20260726, **model_kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_seqs = [sequences[i] for i in train_indices]
    train_labs = [labels[i] for i in train_indices]
    test_seqs = [sequences[i] for i in test_indices]
    test_labs = [labels[i] for i in test_indices]

    train_loader = DataLoader(SequenceDataset(train_seqs, train_labs), batch_size=32, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(SequenceDataset(test_seqs, test_labs), batch_size=64, shuffle=False, collate_fn=pad_collate)

    model = model_cls(input_dim=input_dim, **model_kwargs)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    start_time = time.perf_counter()
    model.train()
    for epoch in range(15):
        for seqs, lbls, lens in train_loader:
            optimizer.zero_grad()
            preds = model(seqs, lens)
            loss = criterion(preds, lbls)
            loss.backward()
            optimizer.step()
    training_seconds = time.perf_counter() - start_time

    model.eval()
    start_time = time.perf_counter()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for seqs, lbls, lens in test_loader:
            probs = model(seqs, lens)
            preds = (probs >= 0.5).long()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(lbls.cpu().numpy())
    inference_ms_per_row = (time.perf_counter() - start_time) * 1000 / len(test_indices)

    return {
        "accuracy": round(float(accuracy_score(all_targets, all_preds)), 6),
        "precision": round(float(precision_score(all_targets, all_preds, zero_division=0)), 6),
        "recall": round(float(recall_score(all_targets, all_preds, zero_division=0)), 6),
        "f1": round(float(f1_score(all_targets, all_preds, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(all_targets, all_probs)), 6),
        "training_seconds": round(float(training_seconds), 6),
        "inference_ms_per_row": round(float(inference_ms_per_row), 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/benchmarks/metrics.json"))
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch is not available; skipping sequence models.")
        return

    data = pd.read_csv(args.dataset)
    sequences, labels, groups = build_sequential_data(data)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(sequences, labels, groups))

    input_dim = len(BASE_FEATURES)
    lstm_metrics = train_eval_neural_model(InteractionLSTM, sequences, labels, train_idx, test_idx, input_dim=input_dim, seed=args.seed)
    transformer_metrics = train_eval_neural_model(SequenceTransformer, sequences, labels, train_idx, test_idx, input_dim=input_dim, seed=args.seed)

    print("LSTM Metrics:", lstm_metrics)
    print("Transformer Metrics:", transformer_metrics)

    # Append metrics to metrics.json
    if args.metrics_path.exists():
        metrics_data = json.loads(args.metrics_path.read_text(encoding="utf-8"))
        metrics_data["models"]["lstm"] = lstm_metrics
        metrics_data["models"]["transformer"] = transformer_metrics
        args.metrics_path.write_text(json.dumps(metrics_data, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {args.metrics_path} with LSTM and Transformer benchmarks.")


if __name__ == "__main__":
    main()
