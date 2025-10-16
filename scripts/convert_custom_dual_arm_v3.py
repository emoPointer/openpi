#!/usr/bin/env python3
"""
转换自定义双臂机器人数据集为LeRobot格式 (Parquet输出) - v3

v3版本说明：
- 只保留action的8维：前7维（原始的0-6）+ 第15维（原始的14，夹爪）
- 其余处理方式与v2一致

原始数据：
- 16维关节：前14维是双臂各7个关节，最后2维是头部（需舍弃）
- 夹爪在最后两维（第14、15维）

输出格式：
- LeRobot标准Parquet格式
- 保持原始数据不归一化
- 兼容OpenPI训练管道

"""

import argparse
import shutil
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm

import os
os.environ["HF_LEROBOT_HOME"] = str(Path.home() / ".cache/huggingface/lerobot")
HF_LEROBOT_HOME = Path(os.environ["HF_LEROBOT_HOME"])

def resize_image(image_array, target_size=(224, 224)):
    if image_array.shape[:2] == target_size:
        return image_array
    image = Image.fromarray(image_array)
    resized = image.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(resized)

def process_episode_data(hdf5_file_path: str, format_type: str = "v3"):
    with h5py.File(hdf5_file_path, 'r') as f:
        length = f['observations/images_color/head'].shape[0]
        joint_positions = f['observations/jointstate/q'][:]  # (T, 16)
        joint_velocities = f['observations/jointstate/dq'][:]  # (T, 16)
        left_gripper = f['observations/limx_2f_gripper/left'][:]  # (T, 3)
        right_gripper = f['observations/limx_2f_gripper/right'][:]  # (T, 3)
        head_images = f['observations/images_color/head'][:]
        left_wrist_images = f['observations/images_color/left_wrist'][:]

        # state: 当前帧joint_positions前7维+左夹爪（去掉最后一帧）
        length = length - 1
        joint_position = joint_positions[:-1, :7]  # (T-1, 7)
        gripper_position = left_gripper[:-1, 0:1]  # (T-1, 1)
        actions = np.concatenate([
            joint_positions[1:, :7],
            left_gripper[1:, 0:1]
        ], axis=1)  # (T-1, 8)
        # 图像移除最后一帧
        head_images = head_images[:-1]
        left_wrist_images = left_wrist_images[:-1]
        return {
            'length': length,
            'joint_position': joint_position,
            'gripper_position': gripper_position,
            'actions': actions,
            'head_images': head_images,
            'left_wrist_images': left_wrist_images,
        }

def main(
    data_dir: str,
    output_name: str,
    *,
    format_type: Literal["v3"] = "v3",
    mode: Literal["separate", "combined"] = "separate",
    task_description: str = "Dual-arm manipulation task (v3)",
    push_to_hub: bool = False
):
    repo_id = f"your_username/{output_name}"
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"删除现有数据集: {output_path}")
        shutil.rmtree(output_path)
    data_path = Path(data_dir)
    hdf5_files = sorted(list(data_path.glob("*.hdf5")))
    print(f"找到 {len(hdf5_files)} 个HDF5文件: {[f.name for f in hdf5_files]}")
    if not hdf5_files:
        raise ValueError(f"在 {data_dir} 中没有找到HDF5文件")
    print(f"\n📊 数据格式分析:")
    sample_data = process_episode_data(str(hdf5_files[0]), format_type)
    # state_dim = sample_data['state'].shape[1]
    action_dim = sample_data['actions'].shape[1]
    print(f"\n⚙️ 转换配置:")
    print(f"  - 数据格式: {format_type}")
    print(f"  - 处理模式: {mode}")
    # print(f"  - 状态维度: {state_dim}")
    print(f"  - 动作维度: {action_dim}")
    print(f"  - 任务描述: {task_description}")
    print(f"  - 输出格式: Parquet (LeRobot标准格式)")
    print(f"  - 图像配置: exterior_image_1_left(头部) + exterior_image_2_left(右手腕) + wrist_image_left(左手腕)")
    features = {
        "exterior_image_1_left": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image_left": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "joint_position": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_position"],
        },
        "gripper_position": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_position"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["actions"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="dual_arm_robot",
        fps=30,
        features=features,
        image_writer_threads=40,
        image_writer_processes=20,
    )
    if mode == "separate":
        print(f"\n🔄 分离模式: 处理 {len(hdf5_files)} 个独立episodes")
        total_frames = 0
        for i, hdf5_file in enumerate(hdf5_files):
            print(f"\n处理Episode {i+1}/{len(hdf5_files)}: {hdf5_file.name}")
            try:
                episode_data = process_episode_data(str(hdf5_file), format_type)
                episode_length = episode_data['length']
                for frame_idx in tqdm(range(episode_length), desc=f"Episode {i+1}", leave=False):
                    head_image = resize_image(episode_data['head_images'][frame_idx])
                    left_wrist_image = resize_image(episode_data['left_wrist_images'][frame_idx])
                    frame_data = {
                        "exterior_image_1_left": head_image,
                        "wrist_image_left": left_wrist_image,
                        "joint_position": episode_data['joint_position'][frame_idx].astype(np.float32),
                        "gripper_position": episode_data['gripper_position'][frame_idx].astype(np.float32),
                        "actions": episode_data['actions'][frame_idx].astype(np.float32),
                        "task": f"{task_description}",
                    }
                    dataset.add_frame(frame_data)
                total_frames += episode_length
                print(f"  ✅ Episode {i+1} 已保存, 累计帧数: {total_frames}")
                dataset.save_episode()
            except KeyError as e:
                print(f"⚠️ 警告: 处理文件 {os.path.basename(hdf5_file)} 时遇到 KeyError: {e}。跳过此文件。")
                continue
            except Exception as e:
                print(f"❌ 错误: 处理文件 {os.path.basename(hdf5_file)} 时遇到未知错误: {e}。")
                continue
    elif mode == "combined":
        print(f"\n🔄 合并模式: 将 {len(hdf5_files)} 个文件合并为1个episode")
        total_frames = 0
        for i, hdf5_file in enumerate(hdf5_files):
            print(f"\n处理文件 {i+1}/{len(hdf5_files)}: {hdf5_file.name}")
            episode_data = process_episode_data(str(hdf5_file), format_type)
            episode_length = episode_data['length']
            for frame_idx in tqdm(range(episode_length), desc=f"合并文件 {i+1}", leave=False):
                head_image = resize_image(episode_data['head_images'][frame_idx])
                left_wrist_image = resize_image(episode_data['left_wrist_images'][frame_idx])
                frame_data = {
                    "exterior_image_1_left": head_image,
                    "wrist_image_left": left_wrist_image,
                    "joint_position": episode_data['joint_position'][frame_idx].astype(np.float32),
                    "gripper_position": episode_data['gripper_position'][frame_idx].astype(np.float32),
                    "actions": episode_data['actions'][frame_idx].astype(np.float32),
                    "task": task_description,
                }
                dataset.add_frame(frame_data)
            total_frames += episode_length
            print(f"  ✅ 文件 {i+1} 已添加, 累计帧数: {total_frames}")
        dataset.save_episode()
    print(f"\n✅ 转换完成!")
    print(f"  - 数据格式: {format_type}")
    print(f"  - 总episodes: {len(hdf5_files) if mode == 'separate' else 1}")
    print(f"  - 总帧数: {total_frames}")
    # print(f"  - 状态维度: {state_dim}")
    print(f"  - 动作维度: {action_dim}")
    print(f"  - 特征字段: state, actions, 3个图像")
    print(f"  - 输出路径: {output_path}")
    if push_to_hub:
        print(f"\n📤 推送到Hub: {repo_id}")
        dataset.push_to_hub(repo_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换自定义双臂机器人数据集为LeRobot格式 (v3)")
    parser.add_argument("--data_dir", type=str, required=True, help="包含HDF5文件的目录路径")
    parser.add_argument("--output_name", type=str, required=True, help="输出数据集的名称")
    parser.add_argument("--format_type", type=str, choices=["v3"], default="v3", help="数据格式类型")
    parser.add_argument("--mode", type=str, choices=["separate", "combined"], default="separate", help="处理模式")
    parser.add_argument("--task_description", type=str, default="Place the cubes on the table into the red plate", help="任务描述")
    parser.add_argument("--push_to_hub", default=False, action="store_true", help="推送到Hugging Face Hub")
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        output_name=args.output_name,
        format_type=args.format_type,
        mode=args.mode,
        task_description=args.task_description,
        push_to_hub=args.push_to_hub
    )
