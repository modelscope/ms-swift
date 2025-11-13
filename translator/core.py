import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List

class GRUTranslator(nn.Module):
    """
    底层翻译器模型 (GRU Translator)
    
    接收来自VLM的半结构化自然语言指令和机器人当前的本体状态，
    输出未来H个时间步的连续动作序列。
    """
    def __init__(self,
                 state_dim: int = 14,          # 状态维度 (例如 7个关节角度 + 7个关节速度)
                 action_dim: int = 7,          # 动作维度 (例如 7-DoF 关节指令)
                 horizon: int = 10,            # 预测的未来动作序列长度 (H)
                 command_model: str = 'all-MiniLM-L6-v2', # 句子编码模型
                 command_embedding_dim: int = 384,        # 句子编码后的向量维度
                 state_encoder_hidden_dim: int = 128,     # 状态编码器MLP的隐藏层维度
                 gru_hidden_dim: int = 512,               # GRU的隐藏层维度
                 gru_num_layers: int = 2,                 # GRU的层数
                 output_mlp_hidden_dim: int = 256         # 输出头MLP的隐藏层维度
                ):
        """
        初始化模型的所有组件
        """
        super().__init__()
        
        # 保存维度信息
        self.action_dim = action_dim
        self.horizon = horizon

        # 1. 指令编码器 (Command Encoder)
        # 使用预训练的SentenceTransformer模型，默认不参与训练以加快速度
        self.sentence_transformer = SentenceTransformer(command_model)
        # 冻结参数，使其不参与反向传播
        for param in self.sentence_transformer.parameters():
            param.requires_grad = False

        # 2. 状态编码器 (State Encoder)
        # 一个简单的MLP，用于将低维的状态信息映射到更高维的特征空间
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(state_encoder_hidden_dim, state_encoder_hidden_dim),
            nn.ReLU()
        )
        
        # 计算融合后的输入维度
        fused_input_dim = command_embedding_dim + state_encoder_hidden_dim

        # 3. 核心序列处理器 (GRU Layer)
        # batch_first=True 是一个好习惯，它让输入的维度变为 (batch, seq_len, features)
        self.gru = nn.GRU(
            input_size=fused_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True
        )

        # 4. 输出头 (Output Head)
        # 一个MLP，将GRU的最终隐藏状态解码为扁平化的动作序列
        self.output_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, output_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(output_mlp_hidden_dim, action_dim * horizon)
        )

    def forward(self, commands: List[str], states: torch.Tensor) -> torch.Tensor:
        """
        模型的前向传播过程
        
        Args:
            commands (List[str]): 一个批次的自然语言指令列表，长度为 batch_size。
            states (torch.Tensor): 一个批次的机器人当前状态，形状为 (batch_size, state_dim)。
            
        Returns:
            torch.Tensor: 预测的未来动作序列，形状为 (batch_size, horizon, action_dim)。
        """
        # 确保模型在正确的设备上运行
        device = states.device
        
        # 1. 编码指令
        # sentence_transformer.encode 返回一个numpy数组，需要转为tensor
        # 在torch.no_grad()下进行，因为我们不训练这个部分
        with torch.no_grad():
            command_embeddings = self.sentence_transformer.encode(
                commands, convert_to_tensor=True, device=device
            )
            # 确保是float类型
            command_embeddings = command_embeddings.float()

        # 2. 编码状态
        state_embeddings = self.state_encoder(states)

        # 3. 融合输入
        # 将指令和状态的特征向量沿特征维度拼接起来
        fused_input = torch.cat([command_embeddings, state_embeddings], dim=1)
        
        # 4. GRU处理
        # GRU需要一个序列输入，我们将融合后的向量视为长度为1的序列
        # 形状从 (batch_size, fused_input_dim) -> (batch_size, 1, fused_input_dim)
        gru_input = fused_input.unsqueeze(1)
        
        # GRU的输出是 (output, h_n)
        # output: 每个时间步的隐藏状态
        # h_n: 最终的隐藏状态，形状为 (num_layers, batch_size, hidden_dim)
        _, final_hidden_state = self.gru(gru_input)
        
        # 我们只需要最后一层的最终隐藏状态
        # 形状从 (num_layers, batch_size, hidden_dim) -> (batch_size, hidden_dim)
        gru_output = final_hidden_state[-1]

        # 5. 生成动作
        # 将GRU的输出通过输出头MLP，得到扁平化的动作
        flat_actions = self.output_head(gru_output)
        
        # 6. 重塑输出
        # 将扁平化的动作序列重塑为 (batch_size, horizon, action_dim)
        actions = flat_actions.view(-1, self.horizon, self.action_dim)
        
        return actions

# --- 使用示例 ---
if __name__ == '__main__':
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 实例化模型
    model = GRUTranslator().to(device)
    print(model)
    
    # 2. 准备模拟输入数据
    batch_size = 4
    
    # 模拟来自Qwen-VL的指令
    dummy_commands = [
        "将末端执行器向前移动0.15米",
        "稍微向左平移夹爪",
        "将手臂抬高0.05米，并保持姿态",
        "闭合夹爪至一半位置"
    ]
    
    # 模拟机器人的当前状态 (14个自由度：7个角度 + 7个速度)
    dummy_states = torch.randn(batch_size, 14).to(device)

    # 3. 执行模型前向传播
    print("\n--- Running forward pass ---")
    print(f"Input commands (batch_size={len(dummy_commands)}):")
    for cmd in dummy_commands:
        print(f"  - {cmd}")
    print(f"Input states shape: {dummy_states.shape}")

    predicted_actions = model(dummy_commands, dummy_states)

    # 4. 打印输出结果的形状
    print(f"\nOutput predicted actions shape: {predicted_actions.shape}")
    print(f"Expected shape: ({batch_size}, {model.horizon}, {model.action_dim})")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params / 1e6:.2f} M")