#!/bin/bash

stdbuf -oL ../profile_dataset.py --train_path '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_2_top1_train_data.pkl' --val_path '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_2_top1_valid_data.pkl' | tee content2.profile

stdbuf -oL ../profile_dataset.py --train_path '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_3_top1_train_data.pkl' --val_path '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_3_top1_valid_data.pkl' | tee content3.profile

stdbuf -oL ../profile_dataset.py --train_path '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_4_top1_train_data.pkl' --val_path '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_4_top1_valid_data.pkl' | tee content4.profile

stdbuf -oL ../profile_dataset.py --train_path '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_5_top1_train_data.pkl' --val_path '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_5_top1_valid_data.pkl' | tee content5.profile

stdbuf -oL ../profile_dataset.py --train_path '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_1_top1_train_data.pkl' --val_path '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_1_top1_valid_data.pkl' | tee content1.profile
