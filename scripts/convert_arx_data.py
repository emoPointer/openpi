import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm

HF_CACHE_DIR = Path.home() / ".cache/huggingface/lerobot"
os.environ["HF_LEROBOT_HOME"] = str(HF_CACHE_DIR)
HF_LEROBOT_HOME = Path(os.environ["HF_LEROBOT_HOME"])
print(f"LeRobot数据集将被存储在: {HF_LEROBOT_HOME}")


def resize_image(image_array: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    使用Pillow库将图像数组的大小调整为目标尺寸。
    """
    # --- 核心修正：在这里添加颜色通道转换 (BGR -> RGB) ---
    # numpy数组的最后一个维度是颜色通道，我们将其顺序颠倒
    image_rgb_array = image_array[:, :, ::-1]

    if image_rgb_array.shape[:2] == target_size:
        return image_rgb_array
    
    # 使用转换后正确的RGB数组创建Pillow图像
    image = Image.fromarray(image_rgb_array)
    resized_image = image.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
    return np.array(resized_image)


def get_data_specs(hdf5_file_path: Path) -> tuple[int, int]:
    """
    从HDF5文件中正确读取 'observations/qpos' 和 'action' 的维度。
    """
    print(f"\n正在从 '{hdf5_file_path.name}' 文件中检测数据维度...")
    with h5py.File(hdf5_file_path, 'r') as f:
        required_keys = ['action', 'observations/qpos']
        for key in required_keys:
            if key not in f:
                raise KeyError(f"错误: HDF5文件中缺少必需的源键 '{key}'。")

        action_dim = f['action'].shape[1]
        state_dim = f['observations/qpos'].shape[1]
        
    print(f"  - 成功检测到 'observations/qpos' (源) 维度: {state_dim}")
    print(f"  - 成功检测到 'action' (源) 维度: {action_dim}")
    return state_dim, action_dim


def main(data_dir: str, output_name: str, task_description: str):
    """
    主函数，执行HDF5到Parquet的转换。
    """
    
    repo_id = f"community/{output_name}"
    output_path = HF_LEROBOT_HOME / repo_id
    
    if output_path.exists():
        print(f"\n警告: 输出目录 '{output_path}' 已存在，将进行删除重建。")
        shutil.rmtree(output_path)
    
    data_path = Path(data_dir)
    hdf5_files = sorted(list(data_path.glob("*.hdf5")), key=lambda p: int(p.stem))
    
    if not hdf5_files:
        raise FileNotFoundError(f"错误: 在目录 '{data_dir}' 中没有找到任何 .hdf5 文件。")
        
    print(f"\n成功找到 {len(hdf5_files)} 个HDF5文件，将把它们合并成一个episode。")
    
    state_dim, action_dim = get_data_specs(hdf5_files[0])
    
    # features 只包含核心的、需要进行维度和类型检查的数据
    features = {
        'observation.image': {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        'observation.wrist_image': {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        'observation.state': {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["qpos"],
        },
        'actions': {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
    }
    
    print("\n数据集目标结构定义完成，Parquet列名如下:")
    for key in features:
        print(f"  - {key}")
    print("  - task (作为元数据添加)")
        
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="aloha_2_arms", 
        fps=10,
        features=features,
        image_writer_threads=8,
        image_writer_processes=4,
    )
    
    total_frames = 0
    for hdf5_file in tqdm(hdf5_files, desc="正在处理HDF5文件"):
        try:
            with h5py.File(hdf5_file, 'r') as f:
                right_images_source = f['observations/images/right'][:]
                mid_images_source = f['observations/images/mid'][:]
                state_source = f['observations/qpos'][:]
                action_source = f['action'][:]
                num_frames = action_source.shape[0]

                for frame_idx in range(num_frames):
                    # --- 关键修正：将 'prompt' 键名修改为 'task' ---
                    frame_data = {
                        'observation.image': resize_image(right_images_source[frame_idx]),
                        'observation.wrist_image': resize_image(mid_images_source[frame_idx]),
                        'observation.state': state_source[frame_idx].astype(np.float32),
                        'actions': action_source[frame_idx].astype(np.float32),
                        'task': task_description,
                    }
                    dataset.add_frame(frame_data)
                total_frames += num_frames
        except Exception as e:
            print(f"\n处理文件 '{hdf5_file.name}' 时发生错误: {e}。已跳过此文件。")
            continue
            
    if total_frames > 0:
        print("\n正在保存合并后的episode...")
        dataset.save_episode()
    else:
        print("\n警告: 没有处理任何帧，未生成任何输出。")

    print("\n-------------------------------------------")
    print("✅ 数据转换完成!")
    print(f"  - 输出数据集名称: {output_name}")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - episode数量: 1")
    print(f"  - 数据集保存在: {output_path}")
    print("-------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将HDF5数据集转换为LeRobot格式的Parquet文件。")
    parser.add_argument(
        "--data_dir", 
        type=str,
        default="data/merged_data",
        help="包含源HDF5文件的目录路径。"
    )
    parser.add_argument(
        "--output_name", 
        type=str, 
        default="lemon_plate_dataset",
        help="输出数据集的名称。"
    )
    # --- 关键修正：将命令行参数从 '--prompt' 修改为 '--task_description' ---
    parser.add_argument(
        "--task_description",
        type=str,
        default="Place lemons on a plate",
        help="要添加到每一帧的任务描述文本。"
    )
    
    args = parser.parse_args()
    
    main(args.data_dir, args.output_name, args.task_description)

