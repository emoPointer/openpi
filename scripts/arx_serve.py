#!/usr/bin/env python3

"""
Usage:
python inference_server.py --checkpoint /path/to/your/checkpoint

该脚本作为服务器运行，等待来自ROS客户端的连接和观测数据。
它会加载一次OpenPI策略模型，然后持续监听网络请求，
对接收到的数据进行推理，并将结果（动作序列）返回给客户端。
"""

import socket
import pickle
import struct
import time
import click
import numpy as np

# --- OpenPI/JAX Imports ---
# 确保这些库在服务器环境中可用
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

# ================= 辅助函数 =================
def recv_all(sock, n):
    """一个辅助函数，确保从socket中接收到指定字节数的数据"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def receive_data(conn):
    """从socket连接中接收打包好的数据"""
    # 1. 首先接收4个字节的长度信息
    raw_msglen = recv_all(conn, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # 2. 根据长度信息接收完整的数据包
    serialized_data = recv_all(conn, msglen)
    # 3. 反序列化数据
    return pickle.loads(serialized_data)

def send_data(conn, payload):
    """将数据序列化并打包发送到socket连接"""
    # 1. 使用pickle序列化数据
    serialized_payload = pickle.dumps(payload)
    # 2. 将数据长度打包为4个字节（网络字节序）
    msglen = struct.pack('>I', len(serialized_payload))
    # 3. 先发送长度，再发送数据
    conn.sendall(msglen + serialized_payload)

# ================= 主函数 =================
@click.command()
@click.option("--config_name", default="arx_delta_lora", help="OpenPI中的模型配置名称。")
@click.option("--checkpoint", "-c", default="/home/ZhouZhiqiang/openpi/checkpoints/arx_delta_lora/arx_delta_lora_norm/29999", help="模型检查点文件的路径。")
@click.option("--host", default="0.0.0.0", help="服务器监听的IP地址。'0.0.0.0' 表示监听所有可用接口。")
@click.option("--port", default=8987, type=int, help="服务器监听的端口号。")
def main(config_name, checkpoint, host, port):
    """
    主函数：加载模型，启动服务器，处理推理请求。
    """
    # --- 1. 加载 OpenPI 策略模型 (只在启动时加载一次) ---
    print(f"正在从配置 '{config_name}' 加载策略...")
    try:
        config = _config.get_config(config_name)
        checkpoint_dir = download.maybe_download(checkpoint)
        policy = _policy_config.create_trained_policy(config, checkpoint_dir)
        print("策略模型加载成功！")
    except Exception as e:
        print(f"致命错误：加载策略模型失败: {e}")
        return

    # --- 2. 设置并启动TCP服务器 ---
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 允许地址重用
        s.bind((host, port))
        s.listen()
        print(f"服务器已启动，正在监听 {host}:{port} ...")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"客户端 {addr} 已连接。")
                try:
                    while True:
                        print("--- 等待接收新的客户端数据 ---") # 新增
                        observation_dict = receive_data(conn)
                        if observation_dict is None:
                            print(f"客户端 {addr} 断开连接。")
                            break

                        print(f"--- 成功接收并反序列化数据，准备推理 ---") # 新增
                        
                        # --- 4. 执行策略推理 ---
                        try:
                            t_start = time.time()
                            result = policy.infer(observation_dict)
                            actions = result["actions"] # 期望形状为 (horizon, action_dim)
                            t_end = time.time()
                            print(f"推理成功，耗时 {t_end - t_start:.4f} 秒。动作形状: {actions.shape}")
                            
                            # --- 5. 将结果发送回客户端 ---
                            send_data(conn, actions)

                        except Exception as e:
                            print(f"策略推理失败: {e}")
                            # 可以选择发送一个错误信号回客户端
                            send_data(conn, {"error": str(e)})
                            
                except ConnectionResetError:
                    print(f"客户端 {addr} 强制断开连接。")
                except Exception as e:
                    print(f"与客户端 {addr} 通信时发生错误: {e}")

if __name__ == '__main__':
    main()