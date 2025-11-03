#!/usr/bin/env python3
"""
使用Tron2机械臂重播HDF5文件中的关节状态数据

功能:
- 从HDF5文件中读取关节位置数据 (observations/jointstate/q)
- 使用Tron2控制器在实际机械臂上重播这些动作
- 支持调整播放速度和选择特定时间段
- 提供安全检查和暂停功能

使用方法:
python tron2_replay_hdf5.py --hdf5_file /path/to/data.hdf5 --start_frame 0 --end_frame 100 --execution_time 0.1
"""

import argparse
import time
import logging
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

from tron2_control import Tron2, RobotConfig
import limxsdk.datatypes as datatypes
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [REPLAY] - %(levelname)s - %(message)s')


class HDF5Replayer:
    def __init__(self, config: RobotConfig):
        """
        初始化HDF5重播器
        
        Args:
            config: Tron2机器人配置
        """
        self.config = config
        self.tron2 = Tron2(config)
        logging.info("Tron2机器人控制器初始化完成")
        
    def load_hdf5_data(self, hdf5_file_path: str) -> Tuple[np.ndarray, dict]:
        """
        从HDF5文件中加载关节状态数据
        
        Args:
            hdf5_file_path: HDF5文件路径
            
        Returns:
            joint_positions: 关节位置数据 (T, 16) -> 取前14维
            metadata: 数据集元信息
        """
        if not Path(hdf5_file_path).exists():
            raise FileNotFoundError(f"HDF5文件不存在: {hdf5_file_path}")
            
        with h5py.File(hdf5_file_path, 'r') as f:
            # 读取关节位置数据
            if 'observations/jointstate/q' not in f:
                raise KeyError("HDF5文件中缺少 'observations/jointstate/q' 数据")
                
            joint_positions = f['observations/jointstate/q'][:]  # (T, 16)
            
            # 获取数据维度信息
            total_frames = joint_positions.shape[0]
            joint_dim = joint_positions.shape[1]
            
            logging.info(f"成功加载HDF5数据:")
            logging.info(f"  - 文件路径: {hdf5_file_path}")
            logging.info(f"  - 总帧数: {total_frames}")
            logging.info(f"  - 关节维度: {joint_dim}")
            
            # 只使用前14维关节数据（双臂各7关节，舍弃头部2维）
            if joint_dim >= 14:
                joint_positions_filtered = joint_positions[:, :14]  # (T, 14)
                logging.info(f"  - 使用前14维关节数据 (舍弃头部关节)")
            else:
                raise ValueError(f"关节维度不足: {joint_dim} < 14")
            
            # 检查数据有效性
            if np.any(np.isnan(joint_positions_filtered)) or np.any(np.isinf(joint_positions_filtered)):
                logging.warning("检测到NaN或Inf值，请检查数据质量")
            
            # 数据范围检查
            joint_min = joint_positions_filtered.min(axis=0)
            joint_max = joint_positions_filtered.max(axis=0)
            logging.info(f"  - 关节范围: [{joint_min.min():.3f}, {joint_max.max():.3f}]")
            
            metadata = {
                'total_frames': total_frames,
                'joint_dim': 14,
                'joint_ranges': (joint_min, joint_max),
                'original_joint_dim': joint_dim
            }
            
            return joint_positions_filtered, metadata
    
    def limit_joints(self, actions: np.ndarray) -> bool:
        """
        验证关节位置是否在安全范围内
        
        Args:
            joint_positions: 关节位置数据 (T, 14)
            
        Returns:
            是否通过安全检查
        """
        joint_lower_limits = np.array([-3.0787582, -0.25656302, -2.6511548, -2.5143174, -2.3090662, -0.76969, -1.7104234, 
                                         -3.0787582, -2.9077188, -1.4538594, -2.5143174, -0.76969, -0.76969, -1.7104234])
        joint_upper_limits = np.array([2.5485292, 2.9077188, 1.4538594, 2.5143174, 0.76969, 0.76969, 1.7104234, 
                                         2.5485292, 0.25656302, 2.6511548, 2.5143174, 2.3090662, 0.76969, 1.7104234])
        for i in range(actions.shape[0]):  # 遍历每一行（时间步）
            for j in range(14):  # 遍历每个关节
                if actions[i, j] < joint_lower_limits[j]:
                    print(f"警告：第{i}行第{j}维度超出下限 {actions[i, j]:.6f} < {joint_lower_limits[j]:.6f}")
                elif actions[i, j] > joint_upper_limits[j]:
                    print(f"警告：第{i}行第{j}维度超出上限 {actions[i, j]:.6f} > {joint_upper_limits[j]:.6f}")
        
        # 应用限幅
        actions[:, :14] = np.clip(actions[:, :14], joint_lower_limits, joint_upper_limits)
        return actions
    
    def replay_trajectory(self, 
                         joint_positions: np.ndarray,
                         start_frame: int = 0,
                         end_frame: Optional[int] = None,
                         execution_time: float = 1,
                         dry_run: bool = False,
                         play_frames: int = 0,
                         skip_frames: int = 0) -> bool:
        """
        重播关节轨迹
        
        Args:
            joint_positions: 关节位置数据 (T, 14)
            start_frame: 开始帧
            end_frame: 结束帧 (None表示到最后)
            execution_time: 每帧执行时间 (秒)
            dry_run: 是否只是测试，不实际执行
            play_frames: 分块批量执行的块大小 (0表示逐帧执行，>0表示每块包含的帧数)
            skip_frames: (已弃用) 保留参数以兼容旧代码
            
        Returns:
            重播是否成功
        """
        # 建立交互式绘图以跟踪前14维动作
        plt.ion()
        fig, ax = plt.subplots()
        action_lines = [ax.plot([], [], label=f"Joint {i + 1}")[0] for i in range(14)]
        ax.set_title("Action Commands (First 14 Dimensions)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Action Value")
        ax.legend(loc="upper right", ncols=2)
        plt.show(block=False)
        action_history = []
        
        total_frames = joint_positions.shape[0]
        # 处理帧范围
        if end_frame is None:
            end_frame = total_frames
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)
        
        if start_frame >= end_frame:
            logging.error(f"无效的帧范围: start={start_frame}, end={end_frame}")
            return False
        
        # 提取要重播的轨迹段
        trajectory = joint_positions[start_frame:end_frame]  # (N, 14)
        replay_frames = trajectory.shape[0]
        
        logging.info(f"开始重播轨迹:")
        logging.info(f"  - 帧范围: {start_frame} 到 {end_frame} (共 {replay_frames} 帧)")
        logging.info(f"  - 执行时间: {execution_time} 秒/帧")
        if play_frames > 0:
            num_chunks = (replay_frames + play_frames - 1) // play_frames
            logging.info(f"  - 分块模式: 每块 {play_frames} 帧，共 {num_chunks} 块")
            logging.info(f"  - 预计总时间: {replay_frames * execution_time + (num_chunks - 1) * 1:.2f} 秒")
        else:
            logging.info(f"  - 预计总时间: {replay_frames * execution_time:.2f} 秒")
        logging.info(f"  - 测试模式: {'是' if dry_run else '否'}")
        
        if dry_run:
            logging.info("🎯 测试模式：只验证数据，不实际控制机器人")
            for i in range(min(5, replay_frames)):  # 只显示前5帧
                logging.info(f"  - 帧 {start_frame + i}: {trajectory[i]}")
            return True
        
        # 设置灯光效果
        self.tron2.set_robot_light(datatypes.LightEffect.FAST_FLASH_BLUE)
        time.sleep(1)
        
        try:
            # 更新执行时间配置
            original_execution_time = self.config.execution_time
            self.config.execution_time = execution_time

            # 如果指定了 play_frames，则分块批量执行
            if play_frames > 0:
                # 将 trajectory 分成多个块，每块大小为 play_frames
                num_chunks = (replay_frames + play_frames - 1) // play_frames  # 向上取整
                logging.info(f"分块执行模式: 每块 {play_frames} 帧，共 {num_chunks} 块")
                # 为了修正后续块，先复制 trajectory，避免原数据被覆盖
                trajectory_mod = trajectory.copy()
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * play_frames
                    end_idx = min((chunk_idx + 1) * play_frames, replay_frames)
                    chunk_size = end_idx - start_idx
                    # 提取当前块的动作序列
                    action_chunk = trajectory[start_idx:end_idx]  # (chunk_size, 14)
                    logging.info(f"执行第 {chunk_idx + 1}/{num_chunks} 块: 帧 {start_idx} 到 {end_idx-1} (共 {chunk_size} 帧)")
                    # 批量执行当前块
                    action_chunk = self.limit_joints(action_chunk)
                    self.tron2.control_joint(action_chunk)

                    # 收集所有时间步的关节动作用于绘图
                    for step_actions in action_chunk[:, :14]:
                        action_history.append(step_actions.copy())
                    history_array = np.asarray(action_history)
                    timesteps = np.arange(history_array.shape[0])
                    for idx, line in enumerate(action_lines):
                        line.set_data(timesteps, history_array[:, idx])
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    plt.pause(0.001)

                    # 每块执行完后 sleep
                    if chunk_idx < num_chunks - 1:
                        logging.info(f"第 {chunk_idx + 1} 块执行完成，sleep 1 秒")
                        # time.sleep(0.1)
                        # 获取当前机器人状态，修正下一块
                        if not dry_run:
                            robot_state = self.tron2.get_state()
                            if robot_state is not None and hasattr(robot_state, 'q'):
                                current_q = np.array(robot_state.q[:14])
                                expected_q = action_chunk[-1]
                                diff = current_q - expected_q
                                diff *= 1
                                logging.info(f"修正下一块: 当前q-期望q={diff}")
                                # 修正下一块
                                next_start = end_idx
                                next_end = min(next_start + play_frames, replay_frames)
                                if next_start < replay_frames:
                                    trajectory[:] += diff
                            else:
                                logging.warning("无法获取机器人状态，跳过修正")
            else:
                # 不分块，逐帧执行（原逻辑）
                for i, joint_pos in enumerate(trajectory):
                    current_frame = start_frame + i
                    logging.info(f"执行帧 {current_frame}/{end_frame-1} ({i+1}/{replay_frames})")
                    # 创建单帧动作序列
                    action_sequence = joint_pos.reshape(1, -1)  # (1, 14)
                    # 执行控制
                    self.tron2.control_joint(action_sequence)

            # 恢复原始执行时间
            self.config.execution_time = original_execution_time
            # 设置完成灯光效果
            self.tron2.set_robot_light(datatypes.LightEffect.SLOW_FLASH_GREEN)
            logging.info("✅ 轨迹重播完成")
            return True

        except KeyboardInterrupt:
            logging.warning("⚠️ 用户中断重播")
            self.tron2.set_robot_light(datatypes.LightEffect.FAST_FLASH_RED)
            return False
        except Exception as e:
            logging.error(f"❌ 重播过程中发生错误: {e}")
            self.tron2.set_robot_light(datatypes.LightEffect.FAST_FLASH_RED)
            return False
    
    def get_current_robot_state(self) -> Optional[np.ndarray]:
        """
        获取当前机器人状态
        
        Returns:
            当前关节位置 (14,) 或 None
        """
        robot_state = self.tron2.get_state()
        if robot_state and hasattr(robot_state, 'q'):
            current_q = np.array(robot_state.q[:14])  # 取前14维
            return current_q
        return None


def main():
    parser = argparse.ArgumentParser(description="使用Tron2机械臂重播HDF5关节数据")
    parser.add_argument("--hdf5_file", type=str, default="/media/chenzh/A23ECE403ECE0D6B/1/data_collector/data_collector_test/episode_01_2025-10-12-16-20-32.hdf5", required=False, 
                       help="HDF5文件路径")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="开始帧索引 (默认: 0)")
    parser.add_argument("--end_frame", type=int, default=None,
                       help="结束帧索引 (默认: 全部)")
    parser.add_argument("--execution_time", type=float, default=0.03,
                       help="每帧执行时间 (秒, 默认: 0.1)")
    parser.add_argument("--play_frames", type=int, default=48,
                       help="间隔跳跃模式: 连续播放的帧数 (0表示不使用间隔模式，默认: 0)")
    parser.add_argument("--skip_frames", type=int, default=0,
                       help="间隔跳跃模式: 每次跳跃的帧数 (仅在 --play_frames > 0 时有效，默认: 0)")
    parser.add_argument("--robot_ip", type=str, default="10.192.1.2",
                       help="机器人IP地址 (默认: 10.192.1.2)")
    parser.add_argument("--robot_accid", type=str, default="DACH_TRON2A_003",
                       help="机器人序列号 (默认: DACH_TRON2A_003)")
    parser.add_argument("--dry_run", action="store_true",
                       help="测试模式：只验证数据，不实际控制机器人")
    parser.add_argument("--skip_safety_check", action="store_true",
                       help="跳过关节限制安全检查")
    
    args = parser.parse_args()
    
    # 创建机器人配置
    robot_config = RobotConfig(
        ip_address=args.robot_ip,
        accid=args.robot_accid,
        execution_time=args.execution_time
    )
    
    try:
        tron2 = Tron2(robot_config)
        init_position = np.array([
        [0.017199993133544922, 0.43150007724761963, -0.011599842458963394, -1.533500075340271, 0.40090012550354004, 0.0048999786376953125, 0.0024001598358154297, 0.01699972152709961, -0.4277999997138977, 0.018799781799316406, -1.5343998670578003, -0.397599995136261, 0.0058002471923828125, -0.0004995504859834909],
           ])
        tron2.control_joint(init_position)
        # logging.info("进入初始状态")
        print("进入初始状态")
        # 创建重播器
        replayer = HDF5Replayer(robot_config)
        
        # 加载HDF5数据
        logging.info(f"正在加载HDF5文件: {args.hdf5_file}")
        joint_positions, metadata = replayer.load_hdf5_data(args.hdf5_file)
        
        # 安全检查
        # if not args.skip_safety_check:
        #     logging.info("执行安全检查...")
        #     if not replayer.validate_joint_limits(joint_positions):
        #         response = input("检测到超出关节限制的数据，是否继续? (y/N): ")
        #         if response.lower() != 'y':
        #             logging.info("用户取消执行")
        #             return
        
        # 显示当前机器人状态
        current_state = replayer.get_current_robot_state()
        if current_state is not None:
            logging.info(f"当前机器人关节位置: {current_state}")
        
        # 确认执行
        if not args.dry_run:
            logging.info(f"\n准备重播轨迹:")
            logging.info(f"  - 总帧数: {metadata['total_frames']}")
            logging.info(f"  - 重播范围: {args.start_frame} 到 {args.end_frame or metadata['total_frames']}")
            logging.info(f"  - 执行速度: {args.execution_time} 秒/帧")
            
            response = input("\n确认开始重播? (y/N): ")
            if response.lower() != 'y':
                logging.info("用户取消执行")
                return
        
        # 执行重播
        success = replayer.replay_trajectory(
            joint_positions=joint_positions,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            execution_time=args.execution_time,
            dry_run=args.dry_run,
            play_frames=args.play_frames,
            skip_frames=args.skip_frames
        )
        
        if success:
            logging.info("🎉 重播任务完成")
        else:
            logging.error("❌ 重播任务失败")
            
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()