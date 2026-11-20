#!/bin/bash
python train_t5.py --finetune \
  --learning_rate 5e-4 --weight_decay 0.01 \
  --scheduler_type cosine --num_warmup_epochs 1 \
  --max_n_epochs 20 --patience_epochs 4 \
  --batch_size 16 --test_batch_size 64 \
  --experiment_name experiment
