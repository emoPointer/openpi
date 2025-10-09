import jax
import openpi.training.config as _config
from openpi.models.model import Observation, Actions

def analyze_model_checkpoint(config_name: str, checkpoint_path: str):
    """
    加载一个模型的训练配置来分析其输入和输出维度。

    Args:
        config_name: 模型的训练配置名称 (例如 'pi05_droid_finetune').
        checkpoint_path: 到达模型检查点目录的路径.
    """
    print(f"--- 正在分析模型: {config_name} ---")
    print(f"--- 检查点路径: {checkpoint_path} ---\n")

    # 1. 加载训练配置
    try:
        config = _config.get_config(config_name)
    except ValueError as e:
        print(f"错误: 无法找到名为 '{config_name}' 的配置. 请确保名称正确。")
        print(f"详细信息: {e}")
        return

    print("✅ 成功加载训练配置.\n")

    # 2. 从模型配置中获取输入/输出规格
    #   - `inputs_spec()` 方法返回一个元组，包含 Observation 和 Actions 的结构
    #   - 这些结构使用 jax.ShapeDtypeStruct 来描述，它只包含形状和数据类型，而不包含实际数据
    observation_spec, action_spec = config.model.inputs_spec()

    # 3. 打印分析结果
    print("--- 🔬 模型输入 (Observation) 规格 ---")
    print("模型期望的输入数据结构如下:\n")

    # 打印图像输入的规格
    if observation_spec.images:
        print("📷 图像输入 (Images):")
        for name, spec in observation_spec.images.items():
            # 形状通常是 (批次大小, 高度, 宽度, 通道数)
            # 这里我们忽略批次大小，只看单个样本的维度
            print(f"  - '{name}':")
            print(f"      - 形状 (Height, Width, Channels): {spec.shape[1:]}")
            print(f"      - 数据类型: {spec.dtype}")
    else:
        print("📷 图像输入 (Images): 无")

    # 打印机器人状态输入的规格
    if observation_spec.state is not None:
        # 形状通常是 (批次大小, 状态维度)
        state_dim = observation_spec.state.shape[1]
        print(f"\n🤖 机器人状态输入 (State):")
        print(f"  - 维度: {state_dim}")
        print(f"  - 形状: {observation_spec.state.shape[1:]}")
        print(f"  - 数据类型: {observation_spec.state.dtype}")
    else:
        print("\n🤖 机器人状态输入 (State): 无")

    # 打印文本提示输入的规格
    if observation_spec.tokenized_prompt is not None:
        # 形状通常是 (批次大小, 最大Token长度)
        max_token_len = observation_spec.tokenized_prompt.shape[1]
        print(f"\n📝 文本提示输入 (Tokenized Prompt):")
        print(f"  - 最大Token长度: {max_token_len}")
        print(f"  - 形状: {observation_spec.tokenized_prompt.shape[1:]}")
        print(f"  - 数据类型: {observation_spec.tokenized_prompt.dtype}")
    else:
        print("\n📝 文本提示输入 (Tokenized Prompt): 无")

    print("\n" + "="*40 + "\n")

    print("--- 🚀 模型输出 (Actions) 规格 ---")
    # action_spec 的形状通常是 (批次大小, 动作序列长度, 动作维度)
    action_horizon = action_spec.shape[1]
    action_dim = action_spec.shape[2]
    print(f"模型会输出一个动作序列 (Action Sequence):\n")
    print(f"  - 动作维度 (Action Dimension): {action_dim}")
    print(f"  - 动作序列长度 (Action Horizon): {action_horizon}")
    print(f"  - 完整形状 (Batch, Horizon, Dim): {action_spec.shape}")
    print(f"  - 数据类型: {action_spec.dtype}")

    print("\n--- 分析完成 ---")


if __name__ == "__main__":
    # --- 请在这里修改为你自己的配置 ---
    # 使用你微调时用的配置名称
    CONFIG_NAME = "pi05_droid_finetune"
    # 你保存的检查点路径
    CHECKPOINT_PATH = "/home/ZhouZhiqiang/openpi/checkpoints/pi05_droid_finetune/my_experiment/19999"

    analyze_model_checkpoint(CONFIG_NAME, CHECKPOINT_PATH)