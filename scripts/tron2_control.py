from functools import partial
import sys
import threading
import time
import json
import uuid
import logging
from dataclasses import dataclass
from typing import Dict, Any, Iterator
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes

import numpy
import websocket

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CLIENT] - %(levelname)s - %(message)s')

@dataclass
class RobotConfig:
    ip_address: str = "10.192.1.2" 
    accid: str = "DACH_TRON2A_003" #TODO: 替换为您机器人的真实序列号
    control_rate: int = 50
    action_dim: int = 14
    control_horizon: int = 10
    left_wrist_camera_serial: str = "230322270826"  # TODO: 替换为左手腕相机的真实序列号
    right_wrist_camera_serial: str = "230422272089" # TODO: 替换为右手腕相机的真实序列号
    head_camera_serial: str = "343622300603"        # TODO: 替换为头部相机
    left_wrist_camera: bool = True
    right_wrist_camera: bool = True
    head_camera: bool = True
    execution_time: float = 0.07


global ROBOTSTATE
ROBOTSTATE = None

class RobotReceiver:
    def robotStateCallback(self, robot_state: datatypes.RobotState):
        global ROBOTSTATE
        ROBOTSTATE = robot_state

class WebSocketManager:
    def __init__(self, ip_address: str):
        self.ws_url = f"ws://{ip_address}:5000"
        self.ws_client = None
        self.latest_state: Dict[str, Any] = {}
        self.is_connected = False
        
        self.thread = threading.Thread(target=self._run_forever, daemon=True)
        self.thread.start()

    def _on_open(self, ws):
        logging.info(f"成功连接到机器人 WebSocket 服务器 at {self.ws_url}")
        self.is_connected = True

    def _on_message(self, ws, message: str):
        try:
            data = json.loads(message)
            title = data.get("title", "")
            
            if title == "notify_robot_info":
                self.latest_state = data.get("data", {})
            else:
                logging.info(f"收到消息: {message}")
        except json.JSONDecodeError:
            logging.error(f"解析JSON失败: {message}")

    def _on_close(self, ws, close_status_code, close_msg):
        logging.warning(f"连接已关闭: {close_status_code} {close_msg}")
        self.is_connected = False

    def _on_error(self, ws, error):
        logging.error(f"WebSocket 错误: {error}")

    def _run_forever(self):
        """保持 WebSocket 连接"""
        logging.info("正在尝试连接机器人...")
        self.ws_client = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_close=self._on_close,
            on_error=self._on_error
        )
        self.ws_client.run_forever()

    def send_command(self, command: Dict[str, Any]):
        """向机器人发送 JSON 指令"""
        if self.is_connected and self.ws_client:
            try:
                self.ws_client.send(json.dumps(command))
            except Exception as e:
                logging.error(f"发送指令失败: {e}")
        else:
            logging.error("无法发送指令：机器人未连接。")
            
    def get_latest_state(self) -> Dict[str, Any]:
        return self.latest_state

class MoveJSequence:
    def __init__(self, config: RobotConfig, policy_inference_result: numpy.ndarray):    # shape (T, 14)
        self.config = config
        self.policy_inference_result = policy_inference_result
        self.current_step = 0
        
        # expected_shape = (config.control_horizon, config.action_dim)
        # if policy_inference_result.shape != expected_shape:
        #     raise ValueError(f"期望 policy_inference_result 的形状为 {expected_shape}, 但得到 {policy_inference_result.shape}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.current_step = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        # if self.current_step >= self.policy_inference_result.shape[0]:
        if self.current_step >= 16:
            print("step8")
            raise StopIteration
        
        cmd = self.get_single_cmd(self.current_step)
        self.current_step += 1
        return cmd

    def get_single_cmd(self, step: int = 0) -> Dict[str, Any]:
        """为单个步骤生成 movej JSON 指令"""
        if step >= self.policy_inference_result.shape[0]:
            raise IndexError(f"步骤 {step} 超出动作范围 {self.policy_inference_result.shape[0]}")
            
        current_joint_action = self.policy_inference_result[step]

        if current_joint_action.tolist() == [0.017199993133544922, 0.43150007724761963, -0.011599842458963394, -1.533500075340271, 0.40090012550354004, 0.0048999786376953125, 0.0024001598358154297, 0.01699972152709961, -0.4277999997138977, 0.018799781799316406, -1.5343998670578003, -0.397599995136261, 0.0058002471923828125, -0.0004995504859834909]:

            command = {
                "accid": self.config.accid,
                "title": "request_movej",
                "timestamp": int(time.time() * 1000),
                "guid": str(uuid.uuid4()),
                "data": {
                    "time": 2,
                    "joint": current_joint_action.tolist() # 14 joint values in radians
                }
            }

        else:
            command = {
                "accid": self.config.accid,
                "title": "request_movej",
                "timestamp": int(time.time() * 1000),
                "guid": str(uuid.uuid4()),
                "data": {
                    "time": self.config.execution_time,
                    "joint": current_joint_action.tolist() # 14 joint values in radians
                }
            }
        return command
    
class GripperSequence:
    def __init__(self, config: RobotConfig, policy_inference_result: numpy.ndarray):    # shape (T, 16)
        self.config = config
        self.policy_inference_result = policy_inference_result
        self.current_step = 0
        
        # expected_shape = (config.control_horizon, config.action_dim)
        # if policy_inference_result.shape != expected_shape:
        #     raise ValueError(f"期望 policy_inference_result 的形状为 {expected_shape}, 但得到 {policy_inference_result.shape}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.current_step = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        # if self.current_step >= self.policy_inference_result.shape[0]:
        if self.current_step >= 16:
            print("step8")
            raise StopIteration
        
        cmd = self.get_single_gripper_cmd(self.current_step)
        self.current_step += 1
        return cmd

    def get_single_gripper_cmd(self, step: int = 0) -> Dict[str, Any]:
        """为单个步骤生成 gripper JSON 指令"""
        if step >= self.policy_inference_result.shape[0]:
            raise IndexError(f"步骤 {step} 超出动作范围 {self.policy_inference_result.shape[0]}")
            
        current_gripper_action = self.policy_inference_result[step]

        command = {
            "accid": self.config.accid,
            "title": "request_set_limx_2fclaw_cmd",
            "timestamp": int(time.time() * 1000),
            "guid": str(uuid.uuid4()),
            "data": {
                "left_opening": current_gripper_action[0].tolist(),  # 开口度，0-100，无量纲（100对应最小闭合，0对应张开到最大）
                "left_speed": 50,    # 夹爪速度，0~100 无量纲（数值越大速度越快）
                "left_force": 25,   #力，夹爪夹持力，0~100 无单位（数值越大力越大）
                # 如您同时给出以下数据，则会控制右夹爪运动
                "right_opening": current_gripper_action[1].tolist(),  # 开口度，0-100，无量纲（100对应最小闭合，0对应张开到最大）
                "right_speed": 50,    # 夹爪速度，0~100 无量纲（数值越大速度越快）
                "right_force": 25,
            }
        }
        return command

class Tron2:
    def __init__(self, config: RobotConfig):
        
        self.config = config
        self.ws_manager = WebSocketManager(config.ip_address)
        self.robot = Robot(RobotType.Tron2)
        self.robot.init(config.ip_address)
        self.robot_receiver = RobotReceiver()
        self.robotStateCallback = partial(self.robot_receiver.robotStateCallback)
        self.robot.subscribeRobotState(self.robotStateCallback)

        while not self.ws_manager.is_connected:
            time.sleep(0.5)
        
        logging.info("机器人控制实例创建成功！")

    def get_state(self):
        return ROBOTSTATE
    
    def control_joint(self, policy_inference_result: numpy.ndarray ):
        # if execution_time is not None:
        #     self.config.execution_time = execution_time

        try:
            movej_sequence = MoveJSequence(self.config, policy_inference_result)
            for i,cmd in enumerate(movej_sequence):
                self.ws_manager.send_command(cmd)
                time.sleep(self.config.execution_time)  # 等待执行完成
        except Exception as e:
            logging.error(f"发送控制序列失败: {e}")

    def control_gripper(self, policy_inference_result: numpy.ndarray ):
        # if execution_time is not None:
        #     self.config.execution_time = execution_time

        try:
            gripper_sequence = GripperSequence(self.config, policy_inference_result)
            for i,cmd in enumerate(gripper_sequence):
                self.ws_manager.send_command(cmd)
                time.sleep(1)
        except Exception as e:
            logging.error(f"发送夹爪控制序列失败: {e}")

    def control_velocity(self, velocity_sequence: numpy.ndarray, dt: float = None):
        try:
            velocity_sequence = numpy.asarray(velocity_sequence, dtype=float)

            if velocity_sequence.ndim != 2 or velocity_sequence.shape[1] != self.config.action_dim:
                logging.error(f"期望速度输入形状为 (T, {self.config.action_dim}), 实际为 {velocity_sequence.shape}")
                return

            if velocity_sequence.shape[0] == 0:
                logging.warning("速度序列为空，未发送任何指令。")
                return

            dt = dt if dt is not None else 0.1
            if dt <= 0:
                logging.error("dt 必须为正数，速度控制终止。")
                return

            robot_state = self.get_state()
            if robot_state is None:
                logging.error("无法获取机器人当前状态，速度控制终止。")
                return

            current_joint_positions = getattr(robot_state, "q", None)
            if current_joint_positions is None or len(current_joint_positions) < self.config.action_dim:
                logging.error("机器人状态中缺少有效的关节位置，速度控制终止。")
                return

            current_q = numpy.array(current_joint_positions[:self.config.action_dim], dtype=float)
            position_sequence = numpy.empty_like(velocity_sequence)

            for idx, dq in enumerate(velocity_sequence):
                current_q = current_q + dq * dt
                position_sequence[idx] = current_q

            movej_sequence = MoveJSequence(self.config, position_sequence)

            for cmd in movej_sequence:
                self.ws_manager.send_command(cmd)
                time.sleep(dt)
        except Exception as e:
            logging.error(f"发送速度控制序列失败: {e}")

    def control_single_step(self, movej_sequence: MoveJSequence, step: int = 0):
        try:
            cmd = movej_sequence.get_single_cmd(step)
            self.ws_manager.send_command(cmd)
            time.sleep(self.config.execution_time)
        except Exception as e:
            logging.error(f"发送单步控制指令失败: {e}")

    def set_robot_light(self, light_effect: datatypes.LightEffect):
        effect_id = light_effect.value + 1
        
        command = {
            "accid": self.config.accid,
            "title": "request_light_effect",
            "timestamp": int(time.time() * 1000),
            "guid": str(uuid.uuid4()),
            "data": {
                "effect": effect_id
            }
        }
        self.ws_manager.send_command(command)


# 流程就是首先实例化Tron2，通过策略获取动作序列，然后执行control方法
if __name__ == '__main__':
    robot_config = RobotConfig()
    tron2_controller = Tron2(robot_config)
    logging.info("测试灯光，设置灯效为高频绿光...")
    tron2_controller.set_robot_light(datatypes.LightEffect.FAST_FLASH_GREEN)
    time.sleep(2)

    # logging.info("测试速度控制")
    # # 生成一个简单的速度序列，持续5步，每步每个关节速度为0.01
    # velocity_steps = 5
    # velocity_value = 0.01
    # dummy_velocity = numpy.full((velocity_steps, robot_config.action_dim), velocity_value, dtype=float)
    # dummy_velocity = numpy.array([
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.1],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.1],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.1],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.1],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.1],
    # ])
    # # tron2_controller.control_velocity(dummy_velocity, dt=1)
    # logging.info("速度控制序列执行完毕。")
    # time.sleep(2)


    # logging.info("测试绝对位置控制")
    # dummy_policy_output = numpy.array([
    #     [-0.5, 0.3, -0.2, 0.2, 0.2, 0.2, 0.2, -0.5, -0.3, -0.2, 0.2, 0.2, 0.2, 0.2],
    #     [0.106, 0.0414, -0.0462, -0.149, 0.02, 0.11, 0.018, 0.071, -0.035, 0.147, -0.0974, 0.08, 0.23, -0.007],
    #     [-0.5, 0.3, -0.2, 0.2, 0.2, 0.2, 0.2, -0.5, -0.3, -0.2, 0.2, 0.2, 0.2, 0.2],
    #     [0.106, 0.0414, -0.0462, -0.149, 0.02, 0.11, 0.018, 0.071, -0.035, 0.147, -0.0974, 0.08, 0.23, -0.007],
    # ])
    # tron2_controller.control_joint(dummy_policy_output)
    # logging.info("绝对位置控制序列执行完毕。")
    # time.sleep(2)

    logging.info("测试夹爪控制")
    dummy_gripper_output = numpy.array([
        [2, 2], 
        [98, 98],
    ])
    tron2_controller.control_gripper(dummy_gripper_output)
    logging.info("夹爪控制序列执行完毕。")

    logging.info("测试状态获取")
    current_state = tron2_controller.get_state()
    if current_state:
        logging.info(f"获取到机器人状态: {current_state}")
        print(f"Current joint positions (q): {current_state.q[:14]}")
    else:
        logging.warning("未能获取到机器人状态。")

    logging.info("测试程序结束。")