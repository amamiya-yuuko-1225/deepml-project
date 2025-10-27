# python scripts/preprocessing/preprocess_cmdw.py --train_csv data/cmdw/train.csv --test_csv data/cmdw/test.csv --emb_path data/embeddings/crawl-300d-2M.vec --out_prefix data/cmdw_emb

# Data preprocessing
python scripts/preprocessing/preprocess_cmdw.py \
  --train_csv data/cmdw/train.csv \
  --emb_path data/embeddings/crawl-300d-2M.vec \
  --out_prefix data/cmdw_emb \
  --test_size 0.2 \
  --seed 42

# Model training
python scripts/training/train_snli.py