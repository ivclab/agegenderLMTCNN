#!/usr/bin/env sh

python eval.py \
--multitask True \
--model_type LMTCNN \
--eval_dir ./tfrecord/train_val_test_per_fold_agegender/test_fold_is_0 \
--eval_data test \
--model_dir ./models/train_val_test_per_fold_agegender/test_fold_is_0/LMTCNN-run-30707 \
--result_dir ./results \
--batch_size 128