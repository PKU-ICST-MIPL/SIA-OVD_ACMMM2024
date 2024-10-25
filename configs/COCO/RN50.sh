name=$0
. configs/controller.sh

args=" \
--grad_accumulation \
--coco_path /data/wangzishuo/OVD-DATA/coco/ \
--output_dir $work_dir \
--batch_size 2 \
--epochs 35 \
--lr_drop 35 \
--backbone clip_RN50 \
--text_len 15 \
--ovd \
--region_prompt_path /data/wangzishuo/MM24/MM-CORA-region/COCO_RN50_10adapter.pth \
--num_adapters 10 \
--save_every_epoch 5 \
--dim_feedforward 1024 \
--use_nms \
--num_queries 1000 \
--anchor_pre_matching \
--remove_misclassified \
--condition_on_text \
--enc_layers 3 \
--text_dim 1024 \
--condition_bottleneck 128 \
--split_class_p 0.2 \
--model_ema \
--model_ema_decay 0.99996 \
--save_best \
--label_version RN50base \
--disable_init \
--target_class_factor 8 \
"

eval "$header$args$extra_args 2>&1 | tee -a $work_dir/exp_$now.txt"
