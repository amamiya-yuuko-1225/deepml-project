#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT NLI (for Kaggle 'Contradictory, My Dear Watson') in pure PyTorch.

- 80/10/10 stratified split (by language x label) on train.csv
- Pair-encoding via HF tokenizer (e.g., bert-base-multilingual-cased)
- Model: AutoModelForSequenceClassification (num_labels=3)
- AMP/mixed-precision friendly; evaluation reports Val/Test accuracy
- Saves best checkpoint; plots train vs val accuracy
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    get_linear_schedule_with_warmup,
)

from matplotlib import pyplot as plt


# --------------------------
# Utilities
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class EncodedPair:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]  # some models may not use this; we’ll handle gracefully
    label: int


class CMDWDataset(Dataset):
    """
    Build dataset from DataFrame with columns: premise, hypothesis, label
    Uses a multilingual HF tokenizer to encode sentence pairs.
    Encoding is done on-the-fly (lazy); padding is done in collate_fn.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_name: str = "bert-base-multilingual-cased",
        max_len: int = 128,
    ):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        # keep a flag whether the tokenizer creates token_type_ids
        # (e.g., XLM-R doesn't; BERT does)
        test = self.tok("a", "b", truncation=True, max_length=4)
        self.uses_token_type_ids = "token_type_ids" in test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        prem = str(row["premise"])
        hypo = str(row["hypothesis"])
        label = int(row["label"])

        enc = self.tok(
            prem,
            hypo,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,  # will be ignored later if not present
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, torch.Tensor]], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Dynamic padding using tokenizer.pad.
    """
    # tokenizer.pad expects a list of dicts with keys among input_ids, attention_mask, token_type_ids, etc.
    features = []
    labels = []
    for ex in batch:
        feat = {k: v for k, v in ex.items() if k != "label"}
        features.append(feat)
        labels.append(ex["label"])
    padded = tokenizer.pad(
        features,
        padding=True,
        return_tensors="pt",
    )
    padded["labels"] = torch.stack(labels, dim=0)
    return padded


# --------------------------
# Eval
# --------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        total_loss += loss.item() * batch["labels"].size(0)
        total_correct += (logits.argmax(dim=-1) == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return total_loss / max(1, total), total_correct / max(1, total)


# --------------------------
# Train
# --------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    train_path = os.path.join(args.data_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found in {args.data_dir}")

    df = pd.read_csv(train_path)
    # Expecting columns: premise, hypothesis, label, language
    need_cols = {"premise", "hypothesis", "label", "language"}
    if not need_cols.issubset(set(df.columns)):
        raise ValueError(f"train.csv must contain columns: {need_cols}. Found: {set(df.columns)}")

    df = df[["premise", "hypothesis", "label", "language"]].dropna()
    df["label"] = df["label"].astype(int)

    # Stratified split by language x label
    df["strata"] = df["language"].astype(str) + "_" + df["label"].astype(str)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    idx_train, idx_tmp = next(sss1.split(df, df["strata"]))
    df_train = df.iloc[idx_train].copy()
    df_tmp = df.iloc[idx_tmp].copy()

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
    idx_val, idx_test = next(sss2.split(df_tmp, df_tmp["strata"]))
    df_val = df_tmp.iloc[idx_val].copy()
    df_test = df_tmp.iloc[idx_test].copy()

    # Tokenizer / Datasets / Loaders
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    train_ds = CMDWDataset(df_train, tokenizer_name=args.tokenizer, max_len=args.max_len)
    val_ds = CMDWDataset(df_val, tokenizer_name=args.tokenizer, max_len=args.max_len)
    test_ds = CMDWDataset(df_test, tokenizer_name=args.tokenizer, max_len=args.max_len)

    def collate(batch):
        return collate_fn(batch, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        problem_type="single_label_classification",
    ).to(device)

    # Optimizer / Scheduler
    # Common LR for BERT finetuning is 2e-5 ~ 5e-5; weight_decay 0.01 is also common
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Linear warmup then decay (optional but helpful). We'll set warmup to 10% of total steps.
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # AMP
    use_amp = (device.type == "cuda" and args.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # I/O
    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "best_bert.pt")
    best_hf_dir = os.path.join(args.out_dir, "best_hf")
    best_val_acc = 0.0

    train_acc_list = []
    val_acc_list = []

    # Loss already computed internally by HF model if labels passed
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        count = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)  # includes loss & logits
                loss = outputs.loss
                logits = outputs.logits

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = batch["labels"].size(0)
            running_loss += loss.item() * bs
            running_correct += (logits.argmax(dim=-1) == batch["labels"]).sum().item()
            count += bs

            if step % args.log_every == 0:
                print(f"Epoch {epoch} | Step {step}/{len(train_loader)} | "
                      f"Loss {running_loss/max(1,count):.4f} | Acc {running_correct/max(1,count):.4f}")

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        train_acc_list.append(running_correct / max(1, count))
        val_acc_list.append(val_acc)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # save state dict (PyTorch)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                },
                best_path,
            )
            # also save full HF checkpoint to reload with from_pretrained
            os.makedirs(best_hf_dir, exist_ok=True)
            model.save_pretrained(best_hf_dir)
            tokenizer.save_pretrained(best_hf_dir)
            print(f"  -> New best model saved to: {best_path} and HF dir: {best_hf_dir}")

    # Load best and evaluate on held-out test
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state"])
        print(f"Loaded best checkpoint from {best_path}")

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Final Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Best Val Acc: {best_val_acc:.4f}")

    # Plot train vs val accuracy
    plt.figure(figsize=(6, 6))
    plt.plot(range(1, args.epochs + 1), train_acc_list, label="Train Accuracy")
    plt.plot(range(1, args.epochs + 1), val_acc_list, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy (BERT)")
    plt.legend()
    plt.grid()
    fig_path = os.path.join(args.out_dir, "train_val_accuracy_bert.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved accuracy plot to {fig_path}")


def build_argparser():
    p = argparse.ArgumentParser(description="BERT NLI (pure PyTorch) on CMDW")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing train.csv")
    p.add_argument("--out_dir", type=str, default="./checkpoints_bert")
    # Tokenization / truncation
    p.add_argument("--tokenizer", type=str, default="bert-base-multilingual-cased")
    p.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    p.add_argument("--max_len", type=int, default=50)  # to mirror your TF script's SEQ_LEN=50
    # Train
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)  # BERT usually needs a smaller batch
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true", help="Enable CUDA AMP")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=100)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)