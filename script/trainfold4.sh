#!/usr/bin/env sh

python train.py \
--multitask True \
--model_type LMTCNN \
--train_dir ./tfrecord/train_val_test_per_fold_agegender/test_fold_is_3 \
--model_dir ./models \
--optim Momentum \
--eta 0.01 \
--eta_decay_rate 0.1 \
--max_steps 50000 \
--steps_per_decay 10000 \
--batch_size 128