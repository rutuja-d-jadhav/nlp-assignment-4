#!/bin/bash
ARGS="--finetune --weight_decay 0.01 --scheduler_type cosine --num_warmup_epochs 1 --max_n_epochs 20 --patience_epochs 4 --batch_size 16 --test_batch_size 64"
run () {   # $1 folder, $2 learning rate, $3 experiment name, $4 log file
  cd /workspace/nlp-assignment-4/$1
  python train_t5.py $ARGS --learning_rate $2 --experiment_name $3 > /workspace/nlp-assignment-4/$4 2>&1 &
  P=$!
  while [ ! -f records/t5_ft_$3_test.pkl ] && kill -0 $P 2>/dev/null; do sleep 30; done
  sleep 30; kill $P 2>/dev/null
}
run part2_nolt 5e-4 ablation_no_lt ablation_no_lt.log
run part2      1e-4 ablation_lr1e-4 ablation_lr.log
echo ALL DONE > /workspace/nlp-assignment-4/ablations_done.txt
