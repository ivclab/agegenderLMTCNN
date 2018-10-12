#!/usr/bin/env sh

python eval.py \
--multitask=False \
--model_type levi_hassner \
--class_type Age \
--eval_dir ./tfrecord/train_val_test_per_fold_agegender/test_fold_is_1 \
--eval_data test \
--model_dir ./models/train_val_test_per_fold_agegender/test_fold_is_1/levi_hassner-Age-run-5294 \
--result_dir ./results \
--batch_size 128