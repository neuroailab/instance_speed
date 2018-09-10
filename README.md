# instance_task
Reimplement and improve the self-supervised task in "Unsupervised Feature Learning via Non-Parametric Instance Discrimination"

# Code to start the training on GPU:
```
python train_estimator.py --gpu your_gpu_number --exp_id your_exp_id --cache_dir_prefix /path/to/your/model/cache/folder --use_synth
```
On one Titan-Xp, the model can get the speed of around 0.25 second/batch.

# Code to start the training on TPU:
```
python train_estimator.py --tpu_name your_tpu_name --exp_id your_exp_id --cache_dir_prefix /path/to/your/model/cache/folder --use_synth
```
On TPU, the model can get the speed of around 0.5 second/batch.


