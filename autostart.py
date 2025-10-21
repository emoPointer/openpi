#!/usr/bin/env python3
"""
定时执行 train.py 的调度脚本
支持指定具体时间执行或延迟执行
"""

import os
import sys
import time
import subprocess
import datetime
import logging
import json
import re
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)

class TrainScheduler:
    def __init__(self, train_script_path="train.py", gpu_check=False, gpu_memory_threshold=1000, 
                 gpu_utilization_threshold=10, gpu_check_interval=60, gpu_max_wait=None, gpu_id=None, custom_command=None):
        """
        初始化调度器
        :param train_script_path: train.py 文件的路径
        :param gpu_check: 是否启用GPU可用性检查
        :param gpu_memory_threshold: GPU可用内存阈值（MB），低于此值认为GPU忙碌
        :param gpu_utilization_threshold: GPU利用率阈值（%），高于此值认为GPU忙碌
        :param gpu_check_interval: GPU检查间隔时间（秒）
        :param gpu_max_wait: GPU最大等待时间（分钟）
        :param gpu_id: 指定要检查的GPU ID，None表示检查所有GPU
        :param custom_command: 自定义命令字符串，如果提供则忽略train_script_path
        """
        self.train_script_path = Path(train_script_path)
        self.logger = logging.getLogger(__name__)
        self.gpu_check = gpu_check
        self.gpu_memory_threshold = gpu_memory_threshold
        self.gpu_utilization_threshold = gpu_utilization_threshold
        self.gpu_check_interval = gpu_check_interval
        self.gpu_max_wait = gpu_max_wait
        self.gpu_id = gpu_id
        self.custom_command = custom_command
        
    def validate_train_script(self):
        """验证训练脚本或命令是否有效"""
        if self.custom_command:
            # 如果使用自定义命令，只需要检查命令不为空
            if not self.custom_command.strip():
                self.logger.error("自定义命令不能为空!")
                return False
            return True
        else:
            # 检查文件是否存在
            if not self.train_script_path.exists():
                self.logger.error(f"训练脚本 {self.train_script_path} 不存在!")
                return False
            return True
    
    def check_nvidia_smi(self):
        """检查nvidia-smi是否可用"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_gpu_info(self):
        """获取GPU信息"""
        try:
            # 使用nvidia-smi查询GPU信息
            cmd = [
                'nvidia-smi', 
                '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.logger.error("无法获取GPU信息")
                return None
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 6:
                        gpu_info = {
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_used': int(parts[2]),
                            'memory_total': int(parts[3]),
                            'utilization': int(parts[4]),
                            'temperature': int(parts[5])
                        }
                        gpu_info['memory_free'] = gpu_info['memory_total'] - gpu_info['memory_used']
                        gpus.append(gpu_info)
            
            return gpus
            
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            self.logger.error(f"获取GPU信息时发生错误: {str(e)}")
            return None
    
    def is_gpu_available(self):
        """检查指定GPU或所有GPU是否可用"""
        if not self.gpu_check:
            return True
        
        if not self.check_nvidia_smi():
            self.logger.warning("nvidia-smi不可用，跳过GPU检查")
            return True
        
        gpus = self.get_gpu_info()
        if not gpus:
            self.logger.warning("无法获取GPU信息，跳过GPU检查")
            return True
        
        # 如果指定了GPU ID，只检查指定的GPU
        if self.gpu_id is not None:
            target_gpu = None
            for gpu in gpus:
                if gpu['index'] == self.gpu_id:
                    target_gpu = gpu
                    break
            
            if target_gpu is None:
                self.logger.error(f"未找到GPU {self.gpu_id}，可用的GPU: {[gpu['index'] for gpu in gpus]}")
                return False
            
            memory_available = target_gpu['memory_free'] >= self.gpu_memory_threshold
            utilization_ok = target_gpu['utilization'] <= self.gpu_utilization_threshold
            
            self.logger.info(
                f"GPU {target_gpu['index']} ({target_gpu['name']}): "
                f"内存 {target_gpu['memory_free']}/{target_gpu['memory_total']}MB, "
                f"利用率 {target_gpu['utilization']}%, "
                f"温度 {target_gpu['temperature']}°C, "
                f"状态: {'可用' if memory_available and utilization_ok else '忙碌'}"
            )
            
            if memory_available and utilization_ok:
                self.logger.info(f"指定GPU {self.gpu_id} 可用")
                return True
            else:
                self.logger.info(f"指定GPU {self.gpu_id} 忙碌")
                return False
        
        # 原有的检查所有GPU逻辑
        available_gpus = []
        for gpu in gpus:
            memory_available = gpu['memory_free'] >= self.gpu_memory_threshold
            utilization_ok = gpu['utilization'] <= self.gpu_utilization_threshold
            
            if memory_available and utilization_ok:
                available_gpus.append(gpu['index'])
            
            self.logger.info(
                f"GPU {gpu['index']} ({gpu['name']}): "
                f"内存 {gpu['memory_free']}/{gpu['memory_total']}MB, "
                f"利用率 {gpu['utilization']}%, "
                f"温度 {gpu['temperature']}°C, "
                f"状态: {'可用' if memory_available and utilization_ok else '忙碌'}"
            )
        
        if available_gpus:
            self.logger.info(f"发现可用GPU: {available_gpus}")
            return True
        else:
            self.logger.info("所有GPU都在使用中")
            return False
    
    def wait_for_gpu(self, check_interval=60, max_wait_time=None):
        """等待GPU可用"""
        if not self.gpu_check:
            return True
        
        self.logger.info("开始等待GPU可用...")
        start_wait_time = datetime.datetime.now()
        
        while True:
            if self.is_gpu_available():
                self.logger.info("GPU现在可用，继续执行")
                return True
            
            # 检查是否超过最大等待时间
            if max_wait_time:
                elapsed = (datetime.datetime.now() - start_wait_time).total_seconds()
                if elapsed >= max_wait_time * 60:  # max_wait_time是分钟
                    self.logger.error(f"等待GPU超时（{max_wait_time}分钟），放弃执行")
                    return False
            
            self.logger.info(f"GPU忙碌，{check_interval}秒后重新检查...")
            time.sleep(check_interval)
    
    
    def run_train_script(self):
        """执行训练脚本或自定义命令"""
        if not self.validate_train_script():
            return False
        
        # 检查GPU可用性
        if self.gpu_check:
            self.logger.info("检查GPU可用性...")
            if not self.wait_for_gpu(self.gpu_check_interval, self.gpu_max_wait):
                self.logger.error("GPU不可用，取消执行训练脚本")
                return False
            
        try:
            if self.custom_command:
                # 执行自定义命令
                self.logger.info(f"开始执行自定义命令: {self.custom_command}")
                start_time = datetime.datetime.now()
                
                # 使用shell执行完整命令
                result = subprocess.run(
                    self.custom_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd()
                )
            else:
                # 执行原有的Python脚本
                self.logger.info(f"开始执行训练脚本: {self.train_script_path}")
                start_time = datetime.datetime.now()
                
                result = subprocess.run(
                    [sys.executable, str(self.train_script_path)],
                    capture_output=True,
                    text=True,
                    cwd=self.train_script_path.parent
                )
            
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            
            if result.returncode == 0:
                self.logger.info(f"命令执行成功! 耗时: {duration}")
                if result.stdout:
                    self.logger.info(f"输出: {result.stdout}")
            else:
                self.logger.error(f"命令执行失败! 返回码: {result.returncode}")
                if result.stderr:
                    self.logger.error(f"错误信息: {result.stderr}")
                    
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"执行命令时发生异常: {str(e)}")
            return False
    
    def schedule_at_time(self, target_time):
        """
        在指定时间执行训练脚本
        :param target_time: 目标时间，格式 "HH:MM" 如 "14:30"
        """
        try:
            # 解析目标时间
            hour, minute= map(int, target_time.split(':'))
            
            # 获取今天的目标时间
            now = datetime.datetime.now()
            target_datetime = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # 如果目标时间已过，设置为明天
            if target_datetime <= now:
                target_datetime += datetime.timedelta(days=1)
            
            self.logger.info(f"计划在 {target_datetime.strftime('%Y-%m-%d %H:%M:%S')} 执行训练脚本")
            
            # 等待到目标时间
            wait_seconds = (target_datetime - datetime.datetime.now()).total_seconds()
            self.logger.info(f"等待 {wait_seconds:.0f} 秒...")
            
            time.sleep(wait_seconds)
            
            # 执行训练脚本
            self.run_train_script()
            
        except ValueError:
            self.logger.error("时间格式错误! 请使用 HH:MM 格式，如 '14:30'")
        except Exception as e:
            self.logger.error(f"调度执行时发生异常: {str(e)}")
    
    def schedule_after_delay(self, delay_minutes):
        """
        延迟指定分钟后执行训练脚本
        :param delay_minutes: 延迟的分钟数
        """
        try:
            delay_seconds = delay_minutes * 60
            execute_time = datetime.datetime.now() + datetime.timedelta(seconds=delay_seconds)
            
            self.logger.info(f"计划在 {delay_minutes} 分钟后 ({execute_time.strftime('%Y-%m-%d %H:%M:%S')}) 执行训练脚本")
            
            time.sleep(delay_seconds)
            self.run_train_script()
            
        except Exception as e:
            self.logger.error(f"延迟执行时发生异常: {str(e)}")
    
    def schedule_daily(self, target_time):
        """
        每天在指定时间执行训练脚本
        :param target_time: 目标时间，格式 "HH:MM"
        """
        self.logger.info(f"设置每日 {target_time} 执行训练脚本")
        
        while True:
            try:
                self.schedule_at_time(target_time)
                # 执行完成后等待到下一天的同一时间
                time.sleep(60)  # 防止在同一分钟内重复执行
            except KeyboardInterrupt:
                self.logger.info("收到中断信号，停止调度器")
                break
            except Exception as e:
                self.logger.error(f"每日调度发生异常: {str(e)}")
                time.sleep(300)  # 出错后等待5分钟再重试

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练脚本调度器')
    parser.add_argument('--script', '-s', default='train.py', help='训练脚本路径 (默认: train.py)')
    parser.add_argument('--command', '-c', default='CUDA_VISIBLE_DEVICES="0,1,3,4" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py arx_delta_lora --exp-name=arx_without_q --overwrite --batch_size=64', help='自定义命令 (如果指定，将忽略--script参数)')
    parser.add_argument('--time', '-t', help='指定执行时间，格式 HH:MM，如 14:30')
    parser.add_argument('--delay', '-d', type=int, help='延迟执行的分钟数')
    parser.add_argument('--daily', action='store_true', help='每日重复执行')
    parser.add_argument('--now', action='store_true', help='立即执行')
    
    # GPU 相关参数
    parser.add_argument('--gpu-check', action='store_true', help='启用GPU可用性检查')
    parser.add_argument('--gpu-id', type=int, default=None,
                        help='指定要检查的GPU ID (如: 0, 1, 2...)，不指定则检查所有GPU')
    parser.add_argument('--gpu-memory', type=int, default=1000, 
                        help='GPU可用内存阈值(MB)，低于此值认为GPU忙碌 (默认: 1000)')
    parser.add_argument('--gpu-util', type=int, default=20000,
                        help='GPU利用率阈值(%%)，高于此值认为GPU忙碌 (默认: 10)')
    parser.add_argument('--gpu-wait-interval', type=int, default=60,
                        help='GPU检查间隔时间(秒) (默认: 60)')
    parser.add_argument('--gpu-max-wait', type=int, default=None,
                        help='GPU最大等待时间(分钟)，超时后放弃执行 (默认: 无限等待)')
    
    args = parser.parse_args()
    
    scheduler = TrainScheduler(
        train_script_path=args.script,
        gpu_check=args.gpu_check,
        gpu_memory_threshold=args.gpu_memory,
        gpu_utilization_threshold=args.gpu_util,
        gpu_check_interval=args.gpu_wait_interval,
        gpu_max_wait=args.gpu_max_wait,
        gpu_id=args.gpu_id,
        custom_command=args.command
    )
    
    if args.now:
        # 立即执行
        scheduler.run_train_script()
    elif args.time:
        if args.daily:
            # 每日执行
            scheduler.schedule_daily(args.time)
        else:
            # 单次执行
            scheduler.schedule_at_time(args.time)
    elif args.delay:
        # 延迟执行
        scheduler.schedule_after_delay(args.delay)
    else:
        # 没有指定参数，显示帮助
        parser.print_help()
        print("\n使用示例:")
        print("  python scheduler.py --now                               # 立即执行train.py")
        print("  python scheduler.py --time 14:30                       # 今天14:30执行")
        print("  python scheduler.py --time 09:00 --daily               # 每天9:00执行")
        print("  python scheduler.py --delay 30                         # 30分钟后执行")
        print("\n自定义命令示例:")
        print('  python scheduler.py --now --command "uv run scripts/train.py --batch_size=64"')
        print('  python scheduler.py --time 14:30 --command \\')
        print('    "CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py tron2_finetune --exp-name=my_experiment3"')
        print("\nGPU检查示例:")
        print("  python scheduler.py --now --gpu-check                  # 立即执行，检查所有GPU")
        print("  python scheduler.py --now --gpu-check --gpu-id 0      # 立即执行，只检查GPU 0")
        print("  python scheduler.py --time 14:30 --gpu-check \\")
        print("                      --gpu-id 1 --gpu-memory 2000      # 检查GPU 1是否有2GB内存")
        print("  python scheduler.py --time 09:00 --daily --gpu-check \\ ")
        print("                      --gpu-id 0 --gpu-max-wait 120     # 每天9:00执行，等待GPU 0最多2小时")

if __name__ == "__main__":
    main()