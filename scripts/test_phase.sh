CUDA_VISIBLE_DEVICES=0 python downstream_phase/run_phase_training.py \
--batch_size 4 \
--epochs 50 \
--save_ckpt_freq 10 \
--model  surgformer_HTA \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 5 \
--data_path /home/han/Documents/github/halo_project/dataset/dataset_keyframe/ \
--eval_data_path /home/han/Documents/github/halo_project/dataset/dataset_keyframe/ \
--nb_classes 11 \
--data_strategy online \
--output_mode key_frame \
--num_frames 16 \
--sampling_rate 4 \
--eval \
--output_dir /mnt/hdd1/ale/Surgformer/outputs/halo \
--pretrained_path pretrain_params/timesformer_base_patch16_224_K400.pyth \
--log_dir /mnt/hdd1/ale/Surgformer/outputs/halo \
--data_set halo \
--data_fps 1fps \
--num_workers 10 \
--no_auto_resume \
--cut_black

# --finetune /home/yangshu/Surgformer/results/surgformer_base_Cataract101_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/checkpoint-best/mp_rank_00_model_states.pt \
