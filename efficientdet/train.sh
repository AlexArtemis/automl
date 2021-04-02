export CUDA_VISIBLE_DEVICES=3
nohup python3 main.py --mode=train_and_eval \
    --train_file_pattern=/home/leizehua/workspace/data/efficient_tfrecord/train/*.tfrecord \
    --val_file_pattern=/home/leizehua/workspace/data/efficient_tfrecord/val/*.tfrecord \
    --model_name=efficientdet-d0 \
    --model_dir=./model_zoo/efficientdet-d0-3-26  \
    --ckpt=./model_zoo/efficientdet-d0  \
    --train_batch_size=64 \
    --eval_batch_size=64 --eval_samples=1024 \
    --num_examples_per_epoch=6000 --num_epochs=50  \
    --hparams=./config/config.yaml \
    --strategy=gpus &