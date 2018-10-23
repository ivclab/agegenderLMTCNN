#!/usr/bin/env sh

python inference.py \
--multitask=True \
--model_type LMTCNN-1-1 \
--model_dir ./models/train_val_test_per_fold_agegender/test_fold_is_3/LMTCNN-1-1-run-1153 \
--convertpb=True \
--filename ./adiencedb/aligned/7285955@N06/landmark_aligned_face.2050.9486613949_909254ccf9_o.jpg