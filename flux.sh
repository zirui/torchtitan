


#export CUDA_VISIBLE_DEVICES=4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,1,2,3

#export PYTHONPATH=/zirui/code/ALTO:/zirui/code/torchtitan-wh:${PYTHONPATH}
export PYTHONPATH=/zirui/code/ALTO:/zirui/code/torchtitan:${PYTHONPATH}
NGPU=8 \
MODULE=alto.models.flux CONFIG=flux_schnell_lpt_mse_4_6_shifted_schedule_mlperf \
./run_train.sh \
    --dump_folder \
    ./outputs/flux_schnell_lpt_precision_schedule_shifted46_mlperf \
    --metrics.enable_tensorboard \
    --metrics.log_freq 100 \
    --checkpoint.enable \
    --checkpoint.keep_latest_k 3 \
    --dataloader.dataset cc12m-wds \
    --dataloader.dataset_path /zirui/data/cc12m-wds \
    --validator.dataloader.dataset_path /zirui/data/COCO-Text \
    --debug.seed 10556 \
    --training.local_batch_size=32 \
