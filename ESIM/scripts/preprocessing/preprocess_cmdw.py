# -*- coding: utf-8 -*-
"""
Preprocess NLI data with optional stratified split when test set has no labels.

Usage examples:
1) Train/test split from train_csv (no test_csv):
   python preprocess.py --train_csv train.csv --emb_path wiki.vec --out_prefix out_dir

2) Test CSV provided BUT without labels -> still split from train_csv:
   python preprocess.py --train_csv train.csv --test_csv test_no_label.csv --emb_path wiki.vec --out_prefix out_dir

3) Test CSV provided WITH labels -> use the provided test set directly:
   python preprocess.py --train_csv train.csv --test_csv test_with_label.csv --emb_path wiki.vec --out_prefix out_dir

You may adjust --test_size and --seed if needed.
"""

import os
import json
import argparse
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt", quiet=True)

LABEL_MAP_TXT = {"entailment": 0, "neutral": 1, "contradiction": 2, "E": 0, "N": 1, "C": 2}


def load_embeddings_txt(vec_path, needed_words):
    word_vec, dim = {}, None
    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        parts = first.strip().split()
        is_header = len(parts) == 2 and all(p.isdigit() for p in parts)
        if not is_header:
            try:
                w, vec = first.split(" ", 1)
                if w in needed_words:
                    arr = np.fromstring(vec, sep=" ")
                    word_vec[w] = arr
                    dim = arr.shape[0]
            except ValueError:
                pass
        for line in tqdm(f, desc="Reading embeddings"):
            try:
                w, vec = line.split(" ", 1)
            except ValueError:
                continue
            if w in needed_words and w not in word_vec:
                arr = np.fromstring(vec, sep=" ")
                if dim is None:
                    dim = arr.shape[0]
                word_vec[w] = arr
    if dim is None:
        raise RuntimeError(f"Failed to detect embedding dim from {vec_path}")
    return word_vec, dim


def build_vocab_and_embeddings(sentences, vec_path, min_freq=1, specials=("<pad>", "<unk>")):
    counter = Counter()
    for s in tqdm(sentences, desc="Tokenizing(all)"):
        counter.update(word_tokenize(str(s)))
    words = [w for w, c in counter.items() if c >= min_freq]
    needed = set(words) | set(specials)
    word_vec, dim = load_embeddings_txt(vec_path, needed)
    worddict = {specials[0]: 0, specials[1]: 1}
    for w in words:
        if w not in worddict:
            worddict[w] = len(worddict)
    scale = 0.05
    emb = np.random.uniform(-scale, scale, (len(worddict), dim)).astype(np.float32)
    emb[worddict[specials[0]]] = 0.0
    for w, idx in worddict.items():
        if w in word_vec:
            emb[idx] = word_vec[w]
    return worddict, emb


def to_ids_and_pad(sentences, worddict, max_len=None, pad="<pad>", unk="<unk>", cap=60):
    tokenized = [word_tokenize(str(s)) for s in sentences]
    if max_len is None:
        max_len = 0 if not tokenized else min(max(len(t) for t in tokenized), cap)
        max_len = max(max_len, 1)
    pad_id, unk_id = worddict[pad], worddict[unk]
    ids, lengths = [], []
    for toks in tokenized:
        arr = [worddict.get(w, unk_id) for w in toks][:max_len]
        lengths.append(len(arr))
        if len(arr) < max_len:
            arr += [pad_id] * (max_len - len(arr))
        ids.append(arr)
    return np.array(ids, dtype=np.int64), np.array(lengths, dtype=np.int64), max_len


def robust_labels(series: pd.Series) -> np.ndarray:
    col = series
    if pd.api.types.is_object_dtype(col):
        vals = col.map(LABEL_MAP_TXT).astype("Int64").fillna(1)
    else:
        vals = col.astype("Int64").fillna(1)
    return vals.clip(0, 2).astype(np.int32).to_numpy()


def has_valid_labels(df: pd.DataFrame, label_col: str) -> bool:
    """
    Return True if the dataframe has a label column with at least one non-null value.
    """
    if label_col not in df.columns:
        return False
    # Consider string labels or numeric; if all NaN or empty strings -> invalid.
    if df[label_col].isna().all():
        return False
    # If the column exists but all blanks after strip, also treat as no labels.
    if pd.api.types.is_object_dtype(df[label_col]):
        if df[label_col].fillna("").map(lambda x: str(x).strip()).eq("").all():
            return False
    return True


def stratified_split(df: pd.DataFrame, label_col: str, test_size: float, seed: int):
    """
    Simple stratified split without sklearn.
    Keeps label distribution similar between train and test.
    """
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for idx, lab in df[label_col].items():
        groups[lab].append(idx)

    test_indices = []
    for lab, indices in groups.items():
        n = len(indices)
        k = max(1, int(round(n * test_size)))
        sel = rng.choice(indices, size=k, replace=False)
        test_indices.extend(sel.tolist())

    test_mask = df.index.isin(test_indices)
    df_test = df.loc[test_mask].copy()
    df_train = df.loc[~test_mask].copy()
    return df_train, df_test


def main(args):
    os.makedirs(args.out_prefix, exist_ok=True)

    # Columns
    prem_col, hypo_col, label_col = "premise", "hypothesis", "label"

    # Load train CSV (must contain labels)
    train_df_full = pd.read_csv(args.train_csv)

    # Decide test source:
    use_provided_test = False
    test_df = None
    if args.test_csv:
        tmp = pd.read_csv(args.test_csv)
        if has_valid_labels(tmp, label_col):
            use_provided_test = True
            test_df = tmp
        else:
            print("[Info] Provided test_csv has no valid labels. Will split from train_csv.")
    else:
        print("[Info] No test_csv provided. Will split from train_csv.")

    # If no valid labeled test set provided, split from train
    if not use_provided_test:
        if not has_valid_labels(train_df_full, label_col):
            raise ValueError("train_csv must contain a valid label column for stratified split.")
        # Stratified split on the original labels (string or numeric)
        print(f"[Info] Performing stratified split: train={1.0 - args.test_size:.2f}, test={args.test_size:.2f}, seed={args.seed}")
        train_df, test_df = stratified_split(train_df_full, label_col, args.test_size, args.seed)
    else:
        train_df = train_df_full

    # Build vocab/embeddings on ALL sentences seen (train + test)
    all_sents = (
        train_df[prem_col].astype(str).tolist()
        + train_df[hypo_col].astype(str).tolist()
        + test_df[prem_col].astype(str).tolist()
        + test_df[hypo_col].astype(str).tolist()
    )
    worddict, emb = build_vocab_and_embeddings(all_sents, args.emb_path, min_freq=args.min_freq)

    # Convert train to IDs
    s1_tr, l1_tr, max_len = to_ids_and_pad(train_df[prem_col].astype(str).tolist(), worddict, cap=args.cap)
    s2_tr, l2_tr, _       = to_ids_and_pad(train_df[hypo_col].astype(str).tolist(), worddict, max_len=max_len)
    labels_tr = robust_labels(train_df[label_col])

    # Convert test to IDs (labels may exist or not; due to our workflow here they do)
    s1_te, l1_te, _ = to_ids_and_pad(test_df[prem_col].astype(str).tolist(), worddict, max_len=max_len)
    s2_te, l2_te, _ = to_ids_and_pad(test_df[hypo_col].astype(str).tolist(), worddict, max_len=max_len)
    labels_te = robust_labels(test_df[label_col]) if has_valid_labels(test_df, label_col) else None

    # IDs
    train_ids = np.arange(len(s1_tr))
    test_ids  = test_df["id"].to_numpy() if "id" in test_df.columns else np.arange(len(s1_te))

    # Pack blobs
    train_blob = {
        "premises": s1_tr,
        "hypotheses": s2_tr,
        "premises_lengths": l1_tr,
        "hypotheses_lengths": l2_tr,
        "labels": labels_tr,
        "ids": train_ids
    }
    test_blob = {
        "premises": s1_te,
        "hypotheses": s2_te,
        "premises_lengths": l1_te,
        "hypotheses_lengths": l2_te,
        "ids": test_ids
    }
    if labels_te is not None:
        test_blob["labels"] = labels_te

    # Save
    with open(os.path.join(args.out_prefix, "train_data.pkl"), "wb") as f:
        pickle.dump(train_blob, f)
    with open(os.path.join(args.out_prefix, "test_data.pkl"), "wb") as f:
        pickle.dump(test_blob, f)
    with open(os.path.join(args.out_prefix, "vocab.pkl"), "wb") as f:
        pickle.dump(worddict, f)
    with open(os.path.join(args.out_prefix, "embeddings.pkl"), "wb") as f:
        pickle.dump(emb, f)

    meta = {
        "max_len": int(max_len),
        "emb_dim": int(emb.shape[1]),
        "vocab_size": int(emb.shape[0])
    }
    with open(os.path.join(args.out_prefix, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[OK] saved to", args.out_prefix)
    if not use_provided_test:
        print(f"[Info] Split sizes -> train: {len(train_df)}, test: {len(test_df)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="Path to training CSV (must have labels).")
    ap.add_argument("--test_csv", default=None, help="Optional test CSV. If missing labels, will split from train.")
    ap.add_argument("--emb_path", required=True, help="Path to .vec/.txt word embeddings.")
    ap.add_argument("--out_prefix", required=True, help="Output directory.")
    ap.add_argument("--min_freq", type=int, default=1, help="Min token frequency for vocab.")
    ap.add_argument("--test_size", type=float, default=0.1, help="Fraction for test when splitting from train.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    ap.add_argument("--cap", type=int, default=60, help="Max cap for sequence length.")
    args = ap.parse_args()
    main(args)


# import os
# import json
# import argparse
# import pickle
# from collections import Counter

# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import nltk
# from nltk.tokenize import word_tokenize
# nltk.download("punkt", quiet=True)

# LABEL_MAP_TXT = {"entailment": 0, "neutral": 1, "contradiction": 2, "E": 0, "N": 1, "C": 2}

# def load_embeddings_txt(vec_path, needed_words):
#     word_vec, dim = {}, None
#     with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
#         first = f.readline()
#         parts = first.strip().split()
#         is_header = len(parts) == 2 and all(p.isdigit() for p in parts)
#         if not is_header:
#             try:
#                 w, vec = first.split(" ", 1)
#                 if w in needed_words:
#                     arr = np.fromstring(vec, sep=" ")
#                     word_vec[w] = arr
#                     dim = arr.shape[0]
#             except ValueError:
#                 pass
#         for line in tqdm(f, desc="Reading embeddings"):
#             try:
#                 w, vec = line.split(" ", 1)
#             except ValueError:
#                 continue
#             if w in needed_words and w not in word_vec:
#                 arr = np.fromstring(vec, sep=" ")
#                 if dim is None:
#                     dim = arr.shape[0]
#                 word_vec[w] = arr
#     if dim is None:
#         raise RuntimeError(f"Failed to detect embedding dim from {vec_path}")
#     return word_vec, dim

# def build_vocab_and_embeddings(sentences, vec_path, min_freq=1, specials=("<pad>", "<unk>")):
#     counter = Counter()
#     for s in tqdm(sentences, desc="Tokenizing(all)"):
#         counter.update(word_tokenize(str(s)))
#     words = [w for w, c in counter.items() if c >= min_freq]
#     needed = set(words) | set(specials)
#     word_vec, dim = load_embeddings_txt(vec_path, needed)
#     worddict = {specials[0]: 0, specials[1]: 1}
#     for w in words:
#         if w not in worddict:
#             worddict[w] = len(worddict)
#     scale = 0.05
#     emb = np.random.uniform(-scale, scale, (len(worddict), dim)).astype(np.float32)
#     emb[worddict[specials[0]]] = 0.0
#     for w, idx in worddict.items():
#         if w in word_vec:
#             emb[idx] = word_vec[w]
#     return worddict, emb

# def to_ids_and_pad(sentences, worddict, max_len=None, pad="<pad>", unk="<unk>", cap=60):
#     tokenized = [word_tokenize(str(s)) for s in sentences]
#     if max_len is None:
#         max_len = 0 if not tokenized else min(max(len(t) for t in tokenized), cap)
#         max_len = max(max_len, 1)
#     pad_id, unk_id = worddict[pad], worddict[unk]
#     ids, lengths = [], []
#     for toks in tokenized:
#         arr = [worddict.get(w, unk_id) for w in toks][:max_len]
#         lengths.append(len(arr))
#         if len(arr) < max_len:
#             arr += [pad_id] * (max_len - len(arr))
#         ids.append(arr)
#     return np.array(ids, dtype=np.int64), np.array(lengths, dtype=np.int64), max_len

# def robust_labels(series: pd.Series) -> np.ndarray:
#     col = series
#     if pd.api.types.is_object_dtype(col):
#         vals = col.map(LABEL_MAP_TXT).astype("Int64").fillna(1)
#     else:
#         vals = col.astype("Int64").fillna(1)
#     return vals.clip(0, 2).astype(np.int32).to_numpy()

# def main(args):
#     os.makedirs(args.out_prefix, exist_ok=True)
#     train_df = pd.read_csv(args.train_csv)
#     test_df = pd.read_csv(args.test_csv)

#     prem_col, hypo_col, label_col = "premise", "hypothesis", "label"

#     all_sents = (
#         train_df[prem_col].astype(str).tolist()
#         + train_df[hypo_col].astype(str).tolist()
#         + test_df[prem_col].astype(str).tolist()
#         + test_df[hypo_col].astype(str).tolist()
#     )
#     worddict, emb = build_vocab_and_embeddings(all_sents, args.emb_path, min_freq=args.min_freq)

#     s1_tr, l1_tr, max_len = to_ids_and_pad(train_df[prem_col].astype(str).tolist(), worddict)
#     s2_tr, l2_tr, _       = to_ids_and_pad(train_df[hypo_col].astype(str).tolist(), worddict, max_len=max_len)
#     labels = robust_labels(train_df[label_col])

#     s1_te, l1_te, _ = to_ids_and_pad(test_df[prem_col].astype(str).tolist(), worddict, max_len=max_len)
#     s2_te, l2_te, _ = to_ids_and_pad(test_df[hypo_col].astype(str).tolist(), worddict, max_len=max_len)

#     train_ids = np.arange(len(s1_tr))
#     test_ids  = test_df["id"].to_numpy() if "id" in test_df.columns else np.arange(len(s1_te))

#     train_blob = {
#         "premises": s1_tr,
#         "hypotheses": s2_tr,
#         "premises_lengths": l1_tr,
#         "hypotheses_lengths": l2_tr,
#         "labels": labels,
#         "ids": train_ids
#     }
#     test_blob = {
#         "premises": s1_te,
#         "hypotheses": s2_te,
#         "premises_lengths": l1_te,
#         "hypotheses_lengths": l2_te,
#         "ids": test_ids
#     }

#     with open(os.path.join(args.out_prefix, "train_data.pkl"), "wb") as f:
#         pickle.dump(train_blob, f)
#     with open(os.path.join(args.out_prefix, "test_data.pkl"), "wb") as f:
#         pickle.dump(test_blob, f)
#     with open(os.path.join(args.out_prefix, "vocab.pkl"), "wb") as f:
#         pickle.dump(worddict, f)
#     with open(os.path.join(args.out_prefix, "embeddings.pkl"), "wb") as f:
#         pickle.dump(emb, f)

#     meta = {"max_len": int(max_len), "emb_dim": int(emb.shape[1]), "vocab_size": int(emb.shape[0])}
#     with open(os.path.join(args.out_prefix, "meta.json"), "w") as f:
#         json.dump(meta, f, indent=2)
#     print("[OK] saved to", args.out_prefix)

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--train_csv", required=True)
#     ap.add_argument("--test_csv", required=True)
#     ap.add_argument("--emb_path", required=True)
#     ap.add_argument("--out_prefix", required=True)
#     ap.add_argument("--min_freq", type=int, default=1)
#     args = ap.parse_args()
#     main(args)