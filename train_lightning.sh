# a lr sgd0.01 adam5e-3
python3 train_lightning.py \
    --batch_size 2 \
    --lr 5e-3 \
    --checkpoint_dir fcn_resnet50_768_ddp \
    --log_dir ./logs/ddp \
    --device_id 0,1 \
    --arch fcn_resnet50 \
    --img_width 768 \
    --use_ddp True