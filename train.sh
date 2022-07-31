
MODEL_FLAGS="--image_size 256 --num_channels 96 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --microbatch 16"


python scripts/image_train.py --log_dir /cluster/scratch/jminder/RoadDiffusion/logs --data_dir /cluster/scratch/jminder/RoadDiffusion/data/ --resume_checkpoint /cluster/scratch/jminder/RoadDiffusion/logs/model035000.pt $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
