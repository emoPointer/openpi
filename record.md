## step1: 将数据集转为lerobot格式
```
python scripts/convert_custom_dual_arm.py --data_dir=/home/ZhouZhiqiang/openpi/data_collector_test --output_name=my_dual_arm_dataset_position --task_description="Place the cubes on the tabl
e into the red plate"
```
## step2: 计算归一化参数
```
uv run scripts/compute_norm_stats.py --config-name tron2_finetune
```
注意tron2_finetune这个配置里面的repo_id要和step1里的对应

## step3: 开始训练
```
CUDA_VISIBLE_DEVICES="0,1,3,4" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py tron2_finetune --exp-name=my_experiment3 --overwrite --batch_size=64
```
训练前需要修改fsdp_devices为对应的数量
CUDA_VISIBLE_DEVICES="5,6" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py tron2_finetune_test --exp-name=my_experiment5 --overwrite --batch_size
=32