python eval.py \
++expt_settings.test_name='drums' \
++checkpoint.resume_path='checkpoint/drums.ckpt' \
++dataset.data_dir='/mnt/lustre/zjtang/data/nerf_synthetic/drums' \
++val.batch_size=12288 \
++ray_param.num_samples=96 \
++model.version=ablenerf \
++dataset.factor=0 \
++model.num_lp=32 \
++model.ff_ratio=3 \
++model.coarse_layers=2 \
++model.fine_layers=6 \