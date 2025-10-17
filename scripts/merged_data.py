import os
import shutil
import random

def move_and_shuffle_hdf5_files():
    """
    将指定文件夹中的HDF5文件移动到一个新文件夹中，并随机打乱顺序进行重命名。
    """
    # --- 1. 定义路径 ---
    # 基础目录
    base_dir = 'arx_data/data'
    # 四个源文件夹的名称
    source_folder_names = [
        'lemon_lower_left',
        'lemon_lower_right',
        'lemon_upper_left',
        'lemon_upper_right'
    ]
    # 目标文件夹的名称
    destination_folder_name = 'merged_data'

    # --- 2. 构造完整的路径 ---
    # 构造源文件夹的完整路径列表
    source_paths = [os.path.join(base_dir, name) for name in source_folder_names]
    # 构造目标文件夹的完整路径
    destination_path = os.path.join(base_dir, destination_folder_name)

    # --- 3. 创建目标文件夹 ---
    # 检查目标文件夹是否存在，如果不存在则创建它
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"成功创建目标文件夹: {destination_path}")
    else:
        print(f"目标文件夹已存在: {destination_path}")

    # --- 4. 收集所有HDF5文件的路径 ---
    all_file_paths = []
    print("\n开始扫描源文件夹...")
    for folder in source_paths:
        # 检查源文件夹是否存在
        if not os.path.isdir(folder):
            print(f"警告: 源文件夹 '{folder}' 不存在，已跳过。")
            continue
        
        # 假设文件名是从 '0.hdf5' 到 '49.hdf5'
        for i in range(50):
            file_name = f"episode_{i}.hdf5"
            file_path = os.path.join(folder, file_name)
            
            # 检查文件是否存在
            if os.path.exists(file_path):
                all_file_paths.append(file_path)
            else:
                print(f"警告: 文件 '{file_path}' 不存在，已跳过。")
    
    total_files = len(all_file_paths)
    if total_files == 0:
        print("错误：未找到任何 .hdf5 文件。请检查你的文件结构和路径。")
        return
        
    print(f"\n扫描完成，总共找到 {total_files} 个文件。")

    # --- 5. 随机打乱文件列表 ---
    random.shuffle(all_file_paths)
    print("文件列表已随机打乱。")

    # --- 6. 移动并重命名文件 ---
    print("\n开始移动并重命名文件...")
    for index, old_path in enumerate(all_file_paths):
        # 新的文件名将是 0.hdf5, 1.hdf5, ..., 199.hdf5
        new_file_name = f"{index}.hdf5"
        new_path = os.path.join(destination_path, new_file_name)
        
        try:
            # 使用 shutil.move 来移动并重命名文件
            shutil.move(old_path, new_path)
        except Exception as e:
            print(f"移动文件 '{old_path}' 时出错: {e}")

    print(f"\n处理完成！所有 {total_files} 个文件已成功移动到 '{destination_path}' 目录中。")


if __name__ == '__main__':
    # 确保你的文件结构与脚本中的 'base_dir' 变量匹配
    # 例如，如果此脚本与 'data' 文件夹在同一级目录，则可以直接运行
    move_and_shuffle_hdf5_files()
