# a lr sgd0.01 adam5e-3
python3 train_new_dev.py \
    --batch_size 2 \
    --lr 0.01 \
    --checkpoint_dir deeplabv3_resnet101_768_0.01 \
    --log_dir ./logs/torchvision \
    --device_id 0 \
    --arch deeplabv3_resnet101 \
    --img_width 768 