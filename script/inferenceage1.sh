#!/usr/bin/env sh

python inference.py \
--multitask=False \
--model_type levi_hassner \
--class_type Age \
--model_dir ./models/train_val_test_per_fold_agegender/test_fold_is_0/levi_hassner-Age-run-6362 \
--convertpb=True \
--filename ./adiencedb/aligned/7285955@N06/landmark_aligned_face.2050.9486613949_909254ccf9_o.jpg