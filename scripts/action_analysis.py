import dataclasses
import jax
import numpy as np
import sys

from openpi.models import model as _model
from openpi.policies import droid_policy, libero_policy, aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

def analyze_model(config_name):
    """分析指定配置的模型"""
    print(f"=== 分析模型: {config_name} ===")
    print()
    
    try:
        # 1. 获取配置信息
        config = _config.get_config(config_name)
        print(f"1. 模型配置:")
        print(f"   - 配置名称: {config_name}")
        print(f"   - 模型类型: {config.model.model_type}")
        print(f"   - Action 维度: {config.model.action_dim}")
        print(f"   - Action 时间窗口: {config.model.action_horizon}")
        print()
        
        # 2. 根据配置类型创建示例数据
        example = None
        example_type = ""
        
        if "droid" in config_name.lower():
            example = droid_policy.make_droid_example()
            example_type = "DROID"
        elif "libero" in config_name.lower():
            example = libero_policy.make_libero_example()
            example_type = "LIBERO"
        elif "aloha" in config_name.lower():
            example = aloha_policy.make_aloha_example()
            example_type = "ALOHA"
        else:
            # 对于其他类型，尝试创建假数据
            fake_obs = config.model.fake_obs()
            fake_act = config.model.fake_act()
            print(f"2. 使用假数据 (无法确定具体类型):")
            print(f"   - 假观察类型: {type(fake_obs)}")
            print(f"   - 假动作类型: {type(fake_act)}")
            print(f"   - 假动作形状: {fake_act.shape}")
            print()
            return config, None, None
            
        print(f"2. {example_type} 示例数据:")
        print(f"   - 示例类型: {type(example)}")
        print(f"   - 示例键: {list(example.keys())}")
        
        # 显示观察数据的详细信息
        for key, value in example.items():
            if hasattr(value, 'shape'):
                print(f"   - {key}: 形状={value.shape}, 类型={type(value)}")
            else:
                print(f"   - {key}: 值='{value}', 类型={type(value)}")
        print()
        
        # 3. 尝试加载模型并进行推理
        checkpoint_name = config_name  # 通常checkpoint名称与config名称相同
        
        # 构建checkpoint路径
        checkpoint_path = f"gs://openpi-assets/checkpoints/{checkpoint_name}"
        print(f"3. 尝试下载checkpoint: {checkpoint_path}")
        
        try:
            checkpoint_dir = download.maybe_download(checkpoint_path)
            policy = _policy_config.create_trained_policy(config, checkpoint_dir)
            
            print("4. 运行推理...")
            result = policy.infer(example)
            
            return config, example, result, policy
            
        except Exception as e:
            print(f"   警告: 无法加载checkpoint ({e})")
            print("   将跳过推理分析")
            return config, example, None, None
            
    except Exception as e:
        print(f"错误: 无法分析配置 {config_name}: {e}")
        return None, None, None, None

def analyze_actions(config, example, result, policy=None):
    """分析action结果"""
    if result is None:
        print("   无推理结果可分析")
        return
        
    # 4. 分析推理结果
    print(f"4. 推理结果分析:")
    print(f"   - result 类型: {type(result)}")
    print(f"   - result 键: {list(result.keys())}")
    print(f"   - actions 类型: {type(result['actions'])}")
    print(f"   - actions 形状: {result['actions'].shape}")
    print(f"   - actions 数据类型: {result['actions'].dtype}")
    print()
    
    # 5. 详细分析action维度
    actions = result["actions"]
    print(f"5. Action 维度详细分析:")
    print(f"   - 完整形状: {actions.shape}")
    print(f"   - 时间步数: {actions.shape[0]} (每次推理预测未来{actions.shape[0]}步)")
    print(f"   - 每步维度: {actions.shape[1]}")
    print()
    
    # 6. 根据不同模型类型分析维度含义
    action_dim = actions.shape[1]
    config_name = config.name.lower()
    
    print(f"6. Action 维度含义分析:")
    if "droid" in config_name:
        if action_dim >= 8:
            print(f"   DROID模型 (8维标准):")
            print(f"   - 维度 0-6: 机器人关节速度 (joint velocity)")
            print(f"   - 维度 7:   夹具位置 (gripper position)")
            if action_dim > 8:
                print(f"   - 维度 8+:  模型内部维度 (实际控制时忽略)")
        else:
            print(f"   DROID模型 ({action_dim}维):")
            print(f"   - 可能是简化版本或不同的action space")
    elif "libero" in config_name:
        print(f"   LIBERO模型 ({action_dim}维):")
        if action_dim == 7:
            print(f"   - 维度 0-6: 机器人关节位置/速度")
            print(f"   - 注意: LIBERO通常不包含独立的gripper维度")
        elif action_dim == 14:
            print(f"   - 维度 0-6: 左臂关节")
            print(f"   - 维度 7-13: 右臂关节")
    elif "aloha" in config_name:
        print(f"   ALOHA模型 ({action_dim}维):")
        if action_dim == 14:
            print(f"   - 维度 0-6: 左臂关节位置")
            print(f"   - 维度 7-13: 右臂关节位置")
        else:
            print(f"   - {action_dim}维: 可能是修改过的ALOHA配置")
    else:
        print(f"   未知模型类型 ({action_dim}维):")
        print(f"   - 需要查看具体配置来确定含义")
    print()
    
    # 7. 显示具体数值（前几个时间步）
    print(f"7. 前几个时间步的action值:")
    for i in range(min(3, actions.shape[0])):
        action_step = actions[i]
        print(f"   时间步 {i}: {action_step}")
    print()
    
    # 8. 分析action的数值范围
    print(f"8. Action 数值范围分析:")
    for i in range(min(action_dim, 10)):  # 最多显示前10个维度
        dim_values = actions[:, i]
        print(f"   维度{i}: min={dim_values.min():.4f}, max={dim_values.max():.4f}, mean={dim_values.mean():.4f}")
    
    if action_dim > 10:
        print(f"   ... (还有 {action_dim - 10} 个维度)")
    print()
    
    # 清理内存
    if policy is not None:
        del policy

# 主程序
if __name__ == "__main__":
    # 如果命令行提供了配置名称，使用它；否则使用默认列表
    if len(sys.argv) > 1:
        config_names = sys.argv[1:]
    else:
        # 默认分析几个常见的配置
        config_names = ["pi0_fast_droid", "pi05_libero", "pi0_aloha_sim"]
    
    for config_name in config_names:
        config, example, result, policy = analyze_model(config_name)
        if config is not None and result is not None:
            analyze_actions(config, example, result, policy)
        print("="*60)
        print()

print("所有分析完成！")

