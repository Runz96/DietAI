#!/bin/bash
#SBATCH --job-name=Nut5K-tr
#SBATCH --partition=h100.q
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=exps/ViT-ENet/test_Nut5K.log

# May 14
#python exec_ENet.py train \
#    --pt_save_path exps/ViT-ENet/MultiTaskElasticNet.pkl

#python exec_ENet.py test \
#    --dataset_path datasets/ASA/ASA24_GPTFoodCodes_largest.csv \
#    --output_path exps/ViT-ENet/ASA24_largest_pred-ViT-ENet.csv 
#python exec_ENet.py test \
#    --dataset_path datasets/ASA/ASA24_GPTFoodCodes_medium.csv \
#    --output_path exps/ViT-ENet/ASA24_medium_pred-ViT-ENet.csv 
#python exec_ENet.py test \
#    --dataset_path datasets/ASA/ASA24_GPTFoodCodes_lowest.csv \
#    --output_path exps/ViT-ENet/ASA24_lowest_pred-ViT-ENet.csv 

python exec_ENet.py test \
    --dataset_path ./datasets/Nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt \
    --dataset_name Nutrition5k \
    --output_path exps/ViT-ENet/Nut5K_test_pred-ViT-ENet.csv