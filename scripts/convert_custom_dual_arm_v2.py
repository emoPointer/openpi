#!/usr/bin/env python3
"""
转换自定义双臂机器人数据集为LeRobot格式 (Parquet输出)

双臂机器人配置:
- 16维关节：前14维是双臂各7个关节，最后2维是头部（需舍弃）
- 双夹爪：左右夹爪独立控制
- 图像：头部相机 + 双手腕相机

输出格式:
- 使用LeRobot标准的Parquet格式存储
- 保持原始数据不进行归一化处理
- 兼容OpenPI训练管道

支持格式:
- droid: 状态=14关节+2夹爪, 动作=14关节位置+2夹爪位置(下一帧)
- libero: 状态=14关节+2夹爪, 动作=14关节位置(下一帧)
- aloha_style: 状态=14关节+2夹爪, 动作=14关节位置+2夹爪位置(下一帧)

注意：最后一帧将被舍弃，因为没有下一帧作为action目标
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

# 设置cache目录
import os
os.environ["HF_LEROBOT_HOME"] = str(Path.home() / ".cache/huggingface/lerobot")
HF_LEROBOT_HOME = Path(os.environ["HF_LEROBOT_HOME"])


def resize_image(image_array, target_size=(224, 224)):
    """调整图像大小到目标尺寸"""
    if image_array.shape[:2] == target_size:
        return image_array
    
    image = Image.fromarray(image_array)
    resized = image.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(resized)


def process_episode_data(hdf5_file_path: str, format_type: str = "droid"):
    """
    处理单个episode数据，支持双臂机器人配置
    
    Args:
        hdf5_file_path: HDF5文件路径
        format_type: 输出格式类型
    
    Returns:
        处理后的episode数据字典
    """
    
    with h5py.File(hdf5_file_path, 'r') as f:
        # 获取数据长度
        length = f['observations/images_color/head'].shape[0]
        
        # 提取原始数据
        joint_positions = f['observations/jointstate/q'][:]  # (T, 16)
        joint_velocities = f['observations/jointstate/dq'][:]  # (T, 16)
        left_gripper = f['observations/limx_2f_gripper/left'][:]  # (T, 3)
        right_gripper = f['observations/limx_2f_gripper/right'][:]  # (T, 3)
        
        # 图像数据
        head_images = f['observations/images_color/head'][:]  # (T, H, W, 3)
        left_wrist_images = f['observations/images_color/left_wrist'][:]  # (T, H, W, 3)
        right_wrist_images = f['observations/images_color/right_wrist'][:]  # (T, H, W, 3)
        
        print(f"  原始数据维度:")
        print(f"    关节位置: {joint_positions.shape} -> 使用前14维 (双臂各7关节，舍弃头部2个自由度)")
        print(f"    关节速度: {joint_velocities.shape} -> 使用前14维")
        print(f"    左夹爪: {left_gripper.shape} -> 使用第0维")
        print(f"    右夹爪: {right_gripper.shape} -> 使用第0维")
        
        # 处理关节数据：使用前14维（双臂各7关节），舍弃头部2维
        joint_pos_filtered = joint_positions[:, :14]  # (T, 14)
        joint_vel_filtered = joint_velocities[:, :14]  # (T, 14)
        
        # 处理夹爪数据
        left_gripper_pos = left_gripper[:, 0]  # (T,) 左夹爪位置
        right_gripper_pos = right_gripper[:, 0]  # (T,) 右夹爪位置
        
        # 数据验证
        assert joint_pos_filtered.shape[1] == 14, f"关节位置维度错误: {joint_pos_filtered.shape[1]}, 应为14"
        assert joint_vel_filtered.shape[1] == 14, f"关节速度维度错误: {joint_vel_filtered.shape[1]}, 应为14"
        assert len(left_gripper_pos) == length, f"左夹爪数据长度不匹配: {len(left_gripper_pos)} vs {length}"
        assert len(right_gripper_pos) == length, f"右夹爪数据长度不匹配: {len(right_gripper_pos)} vs {length}"
        
        # 舍弃最后一帧（因为没有下一帧作为action）
        length = length - 1
        
        # 根据格式类型构建状态和动作
        if format_type == "droid":
            # DROID格式：关节位置和夹爪位置分开存储
            joint_positions = joint_pos_filtered[:-1]  # (T-1, 14) 双臂关节位置，移除最后一帧
            gripper_positions = np.concatenate([
                left_gripper_pos[:-1].reshape(-1, 1),   # 1维左夹爪位置，移除最后一帧
                right_gripper_pos[:-1].reshape(-1, 1)   # 1维右夹爪位置，移除最后一帧
            ], axis=1)  # (T-1, 2)
            
            # 动作：下一帧的关节位置 + 下一帧的夹爪位置
            next_joint_positions = joint_pos_filtered[1:]  # 下一帧的关节位置
            next_left_gripper = left_gripper_pos[1:]  # 下一帧的左夹爪位置
            next_right_gripper = right_gripper_pos[1:]  # 下一帧的右夹爪位置
            
            actions = np.concatenate([
                next_joint_positions,  # 14维下一帧关节位置
                next_left_gripper.reshape(-1, 1),  # 1维下一帧左夹爪位置
                next_right_gripper.reshape(-1, 1)  # 1维下一帧右夹爪位置
            ], axis=1)  # (T-1, 16)
            
            # 同时移除图像数据的最后一帧
            head_images = head_images[:-1]
            left_wrist_images = left_wrist_images[:-1]
            right_wrist_images = right_wrist_images[:-1]
            
            print(f"  DROID格式输出(双臂):")
            print(f"    关节位置维度: {joint_positions.shape[1]} (14关节位置)")
            print(f"    夹爪位置维度: {gripper_positions.shape[1]} (2夹爪位置)")
            print(f"    动作维度: {actions.shape[1]} (14关节位置 + 2夹爪位置 - 下一帧)")
            print(f"    数据长度: {length} (舍弃最后一帧)")
            print(f"    数据范围: 关节[{joint_positions.min():.3f}, {joint_positions.max():.3f}], 夹爪[{gripper_positions.min():.3f}, {gripper_positions.max():.3f}], 动作[{actions.min():.3f}, {actions.max():.3f}]")
            print(f"    ✅ 保持原始数据，无归一化处理")
            
            # 返回分离的数据结构
            return {
                'length': length,
                'joint_positions': joint_positions,
                'gripper_positions': gripper_positions,
                'actions': actions,
                'head_images': head_images,
                'left_wrist_images': left_wrist_images,
                'right_wrist_images': right_wrist_images,
            }
            
        elif format_type == "libero":
            # LIBERO格式：状态包含关节+夹爪，动作为下一帧的关节位置
            states = np.concatenate([
                joint_pos_filtered[:-1],  # 14维关节位置，移除最后一帧
                left_gripper_pos[:-1].reshape(-1, 1),  # 1维左夹爪位置，移除最后一帧
                right_gripper_pos[:-1].reshape(-1, 1)  # 1维右夹爪位置，移除最后一帧
            ], axis=1)  # (T-1, 16)
            
            actions = joint_pos_filtered[1:]  # (T-1, 14) 下一帧的关节位置
            
            # 同时移除图像数据的最后一帧
            head_images = head_images[:-1]
            left_wrist_images = left_wrist_images[:-1]
            right_wrist_images = right_wrist_images[:-1]
            
            print(f"  LIBERO格式输出(双臂):")
            print(f"    状态维度: {states.shape[1]} (14关节位置 + 2夹爪)")
            print(f"    动作维度: {actions.shape[1]} (14关节位置 - 下一帧)")
            print(f"    数据长度: {length} (舍弃最后一帧)")
            print(f"    数据范围: 状态[{states.min():.3f}, {states.max():.3f}], 动作[{actions.min():.3f}, {actions.max():.3f}]")
            print(f"    ✅ 保持原始数据，无归一化处理")
            
        elif format_type == "aloha_style":
            # ALOHA风格：状态包含关节+夹爪，动作包含下一帧关节位置+夹爪位置
            states = np.concatenate([
                joint_pos_filtered[:-1],  # 14维关节位置，移除最后一帧
                left_gripper_pos[:-1].reshape(-1, 1),  # 1维左夹爪位置，移除最后一帧
                right_gripper_pos[:-1].reshape(-1, 1)  # 1维右夹爪位置，移除最后一帧
            ], axis=1)  # (T-1, 16)
            
            # 动作：下一帧的关节位置 + 下一帧的夹爪位置
            next_joint_positions = joint_pos_filtered[1:]  # 下一帧的关节位置
            next_left_gripper = left_gripper_pos[1:]  # 下一帧的左夹爪位置
            next_right_gripper = right_gripper_pos[1:]  # 下一帧的右夹爪位置
            
            actions = np.concatenate([
                next_joint_positions,  # 14维下一帧关节位置
                next_left_gripper.reshape(-1, 1),  # 1维下一帧左夹爪位置
                next_right_gripper.reshape(-1, 1)  # 1维下一帧右夹爪位置
            ], axis=1)  # (T-1, 16)
            
            # 同时移除图像数据的最后一帧
            head_images = head_images[:-1]
            left_wrist_images = left_wrist_images[:-1]
            right_wrist_images = right_wrist_images[:-1]
            
            print(f"  ALOHA风格输出(双臂):")
            print(f"    状态维度: {states.shape[1]} (14关节位置 + 2夹爪)")
            print(f"    动作维度: {actions.shape[1]} (14关节位置 + 2夹爪 - 下一帧)")
            print(f"    数据长度: {length} (舍弃最后一帧)")
            print(f"    数据范围: 状态[{states.min():.3f}, {states.max():.3f}], 动作[{actions.min():.3f}, {actions.max():.3f}]")
            print(f"    ✅ 保持原始数据，无归一化处理")
            
        else:
            raise ValueError(f"不支持的格式类型: {format_type}")
        return {
            'length': length,
            'states': states,
            'actions': actions,
            'head_images': head_images,
            'left_wrist_images': left_wrist_images,
            'right_wrist_images': right_wrist_images,
        }


def main(
    data_dir: str,
    output_name: str,
    *,
    format_type: Literal["droid", "libero", "aloha_style"] = "droid",
    mode: Literal["separate", "combined"] = "separate",
    task_description: str = "Dual-arm manipulation task",
    push_to_hub: bool = False
):
    """
    转换自定义双臂机器人数据集为LeRobot格式
    
    Args:
        data_dir: 包含HDF5文件的目录路径
        output_name: 输出数据集的名称
        format_type: 数据格式类型 ("droid", "libero", "aloha_style")
        mode: 处理模式 ("separate", "combined")
        task_description: 任务描述
        push_to_hub: 是否推送到Hugging Face Hub
    """
    
    # 设置输出路径
    repo_id = f"your_username/{output_name}"
    output_path = HF_LEROBOT_HOME / repo_id
    
    # 清理现有数据集
    if output_path.exists():
        print(f"删除现有数据集: {output_path}")
        shutil.rmtree(output_path)
    
    # 找到所有HDF5文件
    data_path = Path(data_dir)
    hdf5_files = sorted(list(data_path.glob("*.hdf5")))
    print(f"找到 {len(hdf5_files)} 个HDF5文件: {[f.name for f in hdf5_files]}")
    
    if not hdf5_files:
        raise ValueError(f"在 {data_dir} 中没有找到HDF5文件")
    
    # 分析数据维度
    print(f"\n📊 数据格式分析:")
    sample_data = process_episode_data(str(hdf5_files[0]), format_type)
    
    if format_type == "droid":
        joint_dim = sample_data['joint_positions'].shape[1]
        gripper_dim = sample_data['gripper_positions'].shape[1]
        action_dim = sample_data['actions'].shape[1]
    else:
        state_dim = sample_data['states'].shape[1]
        action_dim = sample_data['actions'].shape[1]
    
    print(f"\n⚙️ 转换配置:")
    print(f"  - 数据格式: {format_type}")
    print(f"  - 处理模式: {mode}")
    if format_type == "droid":
        print(f"  - 关节位置维度: {joint_dim}")
        print(f"  - 夹爪位置维度: {gripper_dim}")
        print(f"  - 动作维度: {action_dim}")
    else:
        print(f"  - 状态维度: {state_dim}")
        print(f"  - 动作维度: {action_dim}")
    print(f"  - 任务描述: {task_description}")
    print(f"  - 输出格式: Parquet (LeRobot标准格式)")
    print(f"  - 图像配置: exterior_image_1_left(头部) + exterior_image_2_left(右手腕) + wrist_image_left(左手腕)")
    
    # 根据格式类型定义特征
    if format_type == "droid":
        # DROID格式：分离的关节和夹爪位置
        features = {
            "exterior_image_1_left": {  # 头部相机作为外部相机1
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {  # 右手腕相机作为外部相机2
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {  # 左手腕相机
                "dtype": "image", 
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (joint_dim,),  # 14维双臂关节位置
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (gripper_dim,),  # 2维双夹爪位置
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),  # 16维动作
                "names": ["actions"],
            },
        }
    else:
        # LIBERO/ALOHA格式：合并的状态
        features = {
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image", 
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        }
    
    # 创建LeRobot数据集
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="dual_arm_robot",
        fps=30,
        features=features,
        image_writer_threads=4,
        image_writer_processes=2,
    )
    
    # 处理数据
    if mode == "separate":
        print(f"\n🔄 分离模式: 处理 {len(hdf5_files)} 个独立episodes")
        total_frames = 0
        
        for i, hdf5_file in enumerate(hdf5_files):
            print(f"\n处理Episode {i+1}/{len(hdf5_files)}: {hdf5_file.name}")
            
            episode_data = process_episode_data(str(hdf5_file), format_type)
            episode_length = episode_data['length']
            
            # 使用进度条显示当前episode的处理进度
            for frame_idx in tqdm(range(episode_length), desc=f"Episode {i+1}", leave=False):
                head_image = resize_image(episode_data['head_images'][frame_idx])
                left_wrist_image = resize_image(episode_data['left_wrist_images'][frame_idx])
                right_wrist_image = resize_image(episode_data['right_wrist_images'][frame_idx])
                
                if format_type == "droid":
                    # DROID格式：分离的关节和夹爪位置
                    frame_data = {
                        "exterior_image_1_left": head_image,      # 头部相机作为外部相机1
                        "exterior_image_2_left": right_wrist_image,  # 右手腕相机作为外部相机2
                        "wrist_image_left": left_wrist_image,     # 左手腕相机
                        "joint_position": episode_data['joint_positions'][frame_idx].astype(np.float32),
                        "gripper_position": episode_data['gripper_positions'][frame_idx].astype(np.float32),
                        "actions": episode_data['actions'][frame_idx].astype(np.float32),
                        "task": f"{task_description}",
                    }
                else:
                    # LIBERO/ALOHA格式：合并的状态
                    frame_data = {
                        "image": head_image,
                        "wrist_image": left_wrist_image,
                        "state": episode_data['states'][frame_idx].astype(np.float32),
                        "actions": episode_data['actions'][frame_idx].astype(np.float32),
                        "task": f"{task_description} (execution {i+1})",
                    }
                
                dataset.add_frame(frame_data)
            
            total_frames += episode_length
            print(f"  ✅ Episode {i+1} 已保存, 累计帧数: {total_frames}")
            
            # 保存当前episode
            dataset.save_episode()
    
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
                right_wrist_image = resize_image(episode_data['right_wrist_images'][frame_idx])
                
                if format_type == "droid":
                    # DROID格式：分离的关节和夹爪位置
                    frame_data = {
                        "exterior_image_1_left": head_image,      # 头部相机作为外部相机1
                        "exterior_image_2_left": right_wrist_image,  # 右手腕相机作为外部相机2
                        "wrist_image_left": left_wrist_image,     # 左手腕相机
                        "joint_position": episode_data['joint_positions'][frame_idx].astype(np.float32),
                        "gripper_position": episode_data['gripper_positions'][frame_idx].astype(np.float32),
                        "actions": episode_data['actions'][frame_idx].astype(np.float32),
                        "task": task_description,
                    }
                else:
                    # LIBERO/ALOHA格式：合并的状态
                    frame_data = {
                        "image": head_image,
                        "wrist_image": left_wrist_image,
                        "state": episode_data['states'][frame_idx].astype(np.float32),
                        "actions": episode_data['actions'][frame_idx].astype(np.float32),
                        "task": task_description,
                    }
                
                dataset.add_frame(frame_data)
            
            total_frames += episode_length
            print(f"  ✅ 文件 {i+1} 已添加, 累计帧数: {total_frames}")
        
        # 保存episode
        dataset.save_episode()
    
    
    print(f"\n✅ 转换完成!")
    print(f"  - 数据格式: {format_type}")
    print(f"  - 总episodes: {len(hdf5_files) if mode == 'separate' else 1}")
    print(f"  - 总帧数: {total_frames}")
    if format_type == "droid":
        print(f"  - 关节位置维度: {joint_dim}")
        print(f"  - 夹爪位置维度: {gripper_dim}")
        print(f"  - 动作维度: {action_dim}")
        print(f"  - 特征字段: joint_position, gripper_position, actions, 3个图像")
    else:
        print(f"  - 状态维度: {state_dim}")
        print(f"  - 动作维度: {action_dim}")
        print(f"  - 特征字段: state, actions, 图像")
    print(f"  - 输出路径: {output_path}")
    
    if push_to_hub:
        print(f"\n📤 推送到Hub: {repo_id}")
        dataset.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换自定义双臂机器人数据集为LeRobot格式")
    parser.add_argument("--data_dir", type=str, required=True, help="包含HDF5文件的目录路径")
    parser.add_argument("--output_name", type=str, required=True, help="输出数据集的名称")
    parser.add_argument("--format_type", type=str, choices=["droid", "libero", "aloha_style"], 
                       default="droid", help="数据格式类型")
    parser.add_argument("--mode", type=str, choices=["separate", "combined"], 
                       default="separate", help="处理模式")
    parser.add_argument("--task_description", type=str, 
                       default="Place the cubes on the table into the red plate", help="任务描述")
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