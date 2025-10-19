#!/usr/bin/env python3

"""
Usage:
python inference_ros_client.py --server-ip 192.168.1.100 --frequency 10

该脚本作为ROS节点（客户端）运行。
它订阅机器人和相机话题，将数据打包后通过网络发送到推理服务器，
接收服务器返回的动作序列，然后通过ROS话题发布控制指令。
"""

import sys
import time
from collections import deque
from threading import Lock
import socket
import pickle
import struct

import click
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image

from arm_control.msg import JointInformation
from arm_control.msg import JointControl
# from arm_control.msg import PosCmd # 如果不用可以注释掉

# ================= 全局变量 =================
obs_buffer = deque(maxlen=1)
buffer_lock = Lock()

# ================= 网络通信辅助函数 =================
def recv_all(sock, n):
    """确保从socket中接收到指定字节数的数据"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def send_data(sock, payload):
    """将数据序列化并打包发送到socket"""
    serialized_payload = pickle.dumps(payload)
    msglen = struct.pack('>I', len(serialized_payload))
    sock.sendall(msglen + serialized_payload)

def receive_data(sock):
    """从socket接收打包好的数据"""
    raw_msglen = recv_all(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    serialized_data = recv_all(sock, msglen)
    return pickle.loads(serialized_data)

# ================= ROS & 图像处理函数 =================
def resize_image(image_array, target_size=(224, 224)):
    if image_array.shape[:2] == target_size:
        return image_array
    return cv2.resize(image_array, target_size, interpolation=cv2.INTER_AREA)

def callback(joint_msg, env_img_msg, wrist_msg):
    """
    ROS订阅者的同步回调函数。
    """
    global obs_buffer, buffer_lock
    bridge = CvBridge()
    try:
        # joint_positions = np.array(joint_msg.position[:7], dtype=np.float32)
        joint_positions = np.array(
                [
                    joint_msg.joint_pos[0],
                    joint_msg.joint_pos[1],
                    joint_msg.joint_pos[2],
                    joint_msg.joint_pos[3],
                    joint_msg.joint_pos[4],
                    joint_msg.joint_pos[5],
                    joint_msg.joint_pos[6]
                ]
        )

        env_image_bgr = bridge.imgmsg_to_cv2(env_img_msg, "bgr8")
        wrist_image_bgr = bridge.imgmsg_to_cv2(wrist_msg, "bgr8")
        
        env_image_rgb = cv2.cvtColor(env_image_bgr, cv2.COLOR_BGR2RGB)
        wrist_image_rgb = cv2.cvtColor(wrist_image_bgr, cv2.COLOR_BGR2RGB)

        obs_packet = {
            "joint_positions": joint_positions,
            "env_image": env_image_rgb,
            "wrist_image": wrist_image_rgb,
            "timestamp": joint_msg.header.stamp.to_sec()
        }
        with buffer_lock:
            obs_buffer.append(obs_packet)
    except Exception as e:
        rospy.logerr(f"在回调函数中处理数据时出错: {e}")

@click.command()
@click.option("--server-ip", default="0.0.0.0", help="推理服务器的IP地址。")
@click.option("--server-port", default=8000, type=int, help="推理服务器的端口号。")
@click.option("--frequency", "-f", default=10, type=int, help="控制指令的发布频率 (Hz)。")
def main(server_ip, server_port, frequency):
    """
    主函数：初始化ROS节点，连接到服务器，并运行主控制循环。
    """
    global obs_buffer, buffer_lock

    # --- 1. 初始化ROS节点和发布者/订阅者 ---
    rospy.init_node('openpi_inference_client_node')
    action_pub = rospy.Publisher('follow_joint_control_1', JointControl, queue_size=10)

    joint_sub = Subscriber('joint_information', JointInformation)
    mid_cam_sub = Subscriber("right_camera", Image)# 假设这是手腕相机
    wrist_cam_sub = Subscriber("mid_camera", Image) 

    ats = ApproximateTimeSynchronizer([joint_sub, mid_cam_sub, wrist_cam_sub], queue_size=5, slop=0.1)
    ats.registerCallback(callback)
    rospy.loginfo("ROS节点已初始化，正在订阅话题...")

    # --- 2. 连接到推理服务器 ---
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        rospy.loginfo(f"正在连接到推理服务器 {server_ip}:{server_port} ...")
        client_socket.connect((server_ip, server_port))
        rospy.loginfo("成功连接到服务器！")
    except Exception as e:
        rospy.logfatal(f"连接服务器失败: {e}")
        return

    # --- 3. 主控制循环 ---
    rate = rospy.Rate(frequency)
    
    rospy.loginfo("等待第一次有效的观测数据...")
    while len(obs_buffer) == 0 and not rospy.is_shutdown():
        rospy.sleep(0.5)
    
    rospy.loginfo("已接收到数据，开始主控制循环...")
    
    while not rospy.is_shutdown():
        # 3.1. 从缓冲区获取最新的观测数据
        with buffer_lock:
            if not obs_buffer:
                rospy.logwarn_throttle(5, "观测数据缓冲区为空，跳过此次循环。")
                rate.sleep() # 即使没有数据也要维持循环频率
                continue
            latest_obs = obs_buffer.popleft()

        # 3.2. 准备发送到服务器的观测字典
        try:
            env_image = resize_image(latest_obs['env_image'])
            wrist_image = resize_image(latest_obs['wrist_image'])
            observation_dict = {
                "observation/image": env_image,
                "observation/wrist_image": wrist_image,
                "observation/state": latest_obs['joint_positions'],
                "prompt": "Place lemons on a plate",
            }
        except Exception as e:
            rospy.logerr(f"准备观测字典时出错: {e}")
            continue

        # 3.3. 发送数据并接收推理结果
        try:
            t_start = time.time()
            send_data(client_socket, observation_dict)
            actions = receive_data(client_socket)
            t_end = time.time()
            
            if actions is None:
                rospy.logerr("与服务器的连接已断开，正在退出...")
                break
            if isinstance(actions, dict) and 'error' in actions:
                 rospy.logerr(f"服务器返回推理错误: {actions['error']}")
                 continue

            rospy.loginfo(f"网络通信+推理总耗时 {t_end - t_start:.3f} 秒。获得动作序列形状: {actions.shape}")
        
        except Exception as e:
            rospy.logerr(f"与服务器通信失败: {e}")
            break # 通信失败时，通常需要重新连接，这里我们先直接退出

        # 3.4. 遍历并发布动作序列 (以10Hz频率)
        # 注意: 这里的循环将独占一段时间，直到所有路点都发布完
        for i in range(actions.shape[0]):
            if rospy.is_shutdown():
                break
            
            action_waypoint = actions[i]
            action_msg = JointControl()
            # action_msg.header.stamp = rospy.Time.now()
            action_msg.joint_pos = action_waypoint[:7].tolist()
            
            action_pub.publish(action_msg)
            rate.sleep()

    rospy.loginfo("ROS节点正在关闭。")
    client_socket.close()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print("程序被中断。")