python bert.py \
  --data_dir ./data \
  --out_dir ./checkpoints_bert \
  --tokenizer bert-base-multilingual-cased \
  --model_name bert-base-multilingual-cased \
  --max_len 50 \
  --epochs 10 \
  --batch_size 32 \
  --lr 2e-5 \
  --amp
