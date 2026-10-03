"""
Train and compare 3 alphabet models: MLP, BiLSTM, Transformer.
Saves the best model and produces a comparison report.
"""
import sys, os, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED = PROJECT_ROOT / "data" / "alphabet" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "alphabet"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 31
INPUT_DIM = 126
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load data ──
X_train = np.load(PROCESSED / "X_train.npy")
y_train = np.load(PROCESSED / "y_train.npy")
X_val = np.load(PROCESSED / "X_val.npy")
y_val = np.load(PROCESSED / "y_val.npy")
X_test = np.load(PROCESSED / "X_test.npy")
y_test = np.load(PROCESSED / "y_test.npy")
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64)

# ── Model definitions ──
class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(126, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES),
        )
    def forward(self, x):
        return self.net(x)

class BiGRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(126, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, NUM_CLASSES))
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 126)
        out, _ = self.gru(x)  # (B, 1, 256)
        scores = self.attn(out).squeeze(-1)  # (B, 1)
        weights = F.softmax(scores, dim=-1)  # (B, 1)
        ctx = torch.bmm(weights.unsqueeze(1), out).squeeze(1)  # (B, 256)
        return self.classifier(ctx)

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(126, 64)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 64))
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, NUM_CLASSES))
    def forward(self, x):
        x = self.embed(x.unsqueeze(1))  # (B, 1, 64)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 2, 64)
        x = self.transformer(x)
        cls_out = x[:, 0, :]  # (B, 64)
        return self.classifier(cls_out)

# ── Training function ──
def train_model(model, name, epochs=80, lr=1e-3):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
        val_loss /= len(X_val)
        val_acc = val_correct / len(X_val)

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  [{name}] Epoch {epoch+1:3d} | Loss: {train_loss:.4f}/{val_loss:.4f} | Val Acc: {val_acc:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "history": history},
                       MODELS_DIR / f"{name}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"  [{name}] Early stop at epoch {epoch+1}")
                break

    # Load best and evaluate on test
    ckpt = torch.load(MODELS_DIR / f"{name}_best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_true = [], []
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            test_loss += criterion(logits, yb).item() * len(xb)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(yb.cpu().numpy())
    test_loss /= len(X_test)
    test_acc = accuracy_score(all_true, all_preds)
    test_f1 = f1_score(all_true, all_preds, average="macro")

    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    size_mb = os.path.getsize(MODELS_DIR / f"{name}_best.pth") / (1024*1024)

    # Inference speed
    dummy = torch.randn(100, INPUT_DIM).to(DEVICE)
    model.eval()
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(dummy)
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    latency_ms = (time.time() - t0) / 100 * 1000

    return {
        "name": name,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_loss": test_loss,
        "params": param_count,
        "size_mb": size_mb,
        "latency_ms": latency_ms,
        "history": history,
    }


if __name__ == "__main__":
    # ── Train all models ──
    print("\n" + "=" * 60)
    print("Training MLP")
    print("=" * 60)
    mlp_result = train_model(MLPModel(), "mlp", epochs=80)

    print("\n" + "=" * 60)
    print("Training BiGRU")
    print("=" * 60)
    gru_result = train_model(BiGRUModel(), "bigru", epochs=80)

    print("\n" + "=" * 60)
    print("Training Transformer")
    print("=" * 60)
    trans_result = train_model(TransformerModel(), "transformer", epochs=80)

    # ── Comparison ──
    results = [mlp_result, gru_result, trans_result]
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<15} {'Accuracy':>10} {'F1':>10} {'Params':>10} {'Size':>8} {'Latency':>10}")
    print("-" * 63)
    for r in results:
        print(f"{r['name']:<15} {r['test_acc']:>10.4f} {r['test_f1']:>10.4f} {r['params']:>10,} {r['size_mb']:>7.2f}M {r['latency_ms']:>9.2f}ms")

    # Save comparison
    best = max(results, key=lambda r: r["test_acc"])
    print(f"\nBest model: {best['name']} (accuracy={best['test_acc']:.4f})")

    # Copy best to spell_classifier_best.pth
    import shutil
    src = MODELS_DIR / f"{best['name']}_best.pth"
    dst = MODELS_DIR / "spell_classifier_best.pth"
    shutil.copy2(src, dst)
    print(f"Saved best model to {dst}")

    # Save report
    report = {
        "dataset": "ArSL-31-Pilot",
        "samples": len(X_train) + len(X_val) + len(X_test),
        "num_classes": NUM_CLASSES,
        "splits": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "models": [{k: v for k, v in r.items() if k != "history"} for r in results],
        "best_model": best["name"],
    }
    with open(MODELS_DIR / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {MODELS_DIR / 'comparison_report.json'}")
