#!/bin/bash

# modify data dir for following args
dataroot="<flickr30k_entities_dataset_dir>"
glove_file="<your_glove_6b_txt_dir>"
boxfile="<flickr30k_objects_vocab_txt_dir>"
referoot="<your_refercoco*_dataset_root_dir>"
features_path="<extracted_coco_features_dir>"
mat_root="<features_dir_from_maf>"

# flickr
CUDA_VISIBLE_DEVICES=0 python mytrain.py --epochs 45 \
--glove_file ${glove_file} \
--dataroot ${dataroot} \
--boxfile ${boxfile} \
--referoot ${referoot} \
--features_path ${features_path} \
--mat_root ${mat_root} \
--dropout 0.1 --arch dual_mom --v_feature_dropout_prob 0.1 \
--mask none --batch_size 256 --sum_scale 10 --lr 0.001 --gpu 0 \
--threshold 0.95 --p_t_range 0.1,0.3 --momentum_m 0.99 --falsenegative --seed 42 &


# refcoco*
CUDA_VISIBLE_DEVICES=1 python mytrain.py --epochs 35 \
--glove_file ${glove_file} \
--dataroot ${dataroot} \
--boxfile ${boxfile} \
--referoot ${referoot} \
--features_path ${features_path} \
--mat_root ${mat_root} \
--dropout 0.1 --arch dual_mom --v_feature_dropout_prob 0.1 --task refer --dataset_name \*{refcoco*}*\ \
--mask none --batch_size 256 --sum_scale 10 --lr 0.0005 --gpu 0 \
--threshold 0.95 --p_t_range 0.01,0.5 --momentum_m 0.9 --falsenegative --seed 42 &