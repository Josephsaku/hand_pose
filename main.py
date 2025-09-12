import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# --- 1. 模拟输入数据 ---

vocab = {'<pad>': 0, 'a': 1, 'closed': 2, 'fist': 3, 'open': 4, 'palm': 5}
vocab_size = len(vocab)
max_seq_len = 5  # 假设文本描述最长为5个词

# 文本描述: "a closed fist"
text_description = "a closed fist"
# 将文本转换为 token IDs
token_ids = [vocab.get(word, 0) for word in text_description.split()]
# 填充到最大长度
token_ids += [vocab['<pad>']] * (max_seq_len - len(token_ids))
# 转换为 PyTorch Tensor (batch_size=1, seq_len=5)
text_tensor = torch.LongTensor(token_ids).unsqueeze(0)

# 图像: 创建一个随机的模拟手部姿态图片
# (batch_size=1, channels=3, height=128, width=128)
image_tensor = torch.randn(1, 3, 128, 128)


# --- 2. 定义模型组件 ---

class SimpleImageEncoder(nn.Module):
    """一个简单的CNN，用于从图片中提取特征图"""

    def __init__(self):
        super().__init__()
        # 使用一个简化的卷积网络，实际项目中可用 pre-trained ResNet 等
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 更多层...

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # shape: [batch, 16, 64, 64]
        return x


class SimpleTextEncoder(nn.Module):
    """一个简单的文本编码器，用于将 token ID 转换为词向量"""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)  # shape: [batch, seq_len, embed_dim]


class CrossAttentionModel(nn.Module):
    """
    核心模型：结合图像和文本编码器，并使用 Cross-Attention 进行融合
    """

    def __init__(self, vocab_size, embed_dim=64, num_heads=4):
        super().__init__()
        self.image_encoder = SimpleImageEncoder()
        self.text_encoder = SimpleTextEncoder(vocab_size, embed_dim)

        # 图像特征的维度
        # 经过 SimpleImageEncoder 后, 输出是 [batch, 16, 64, 64]
        # 我们需要把它转换成 MultiheadAttention 能接受的格式
        img_feature_channels = 16
        # 将卷积层的输出通道数调整为与 embed_dim 一致，以便进行注意力计算
        self.img_proj = nn.Conv2d(img_feature_channels, embed_dim, kernel_size=1)

        # PyTorch 内置的 Multi-Head Cross-Attention 模块
        # embed_dim: 文本和图像特征的维度
        # num_heads: 注意力头的数量
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True  # 让输入 batch 维度在第一位
        )

    def forward(self, image, text):
        # 1. 编码图像和文本
        text_features = self.text_encoder(text)  # -> [batch, seq_len, embed_dim]
        image_features_map = self.image_encoder(image)  # -> [batch, 16, 64, 64]

        # 2. 准备 Cross-Attention 的输入 (Query, Key, Value)

        # Query (Q): 来自文本。它代表“我们要问什么/关注什么”。
        # shape: [batch, seq_len, embed_dim]
        query = text_features

        # Key (K) 和 Value (V): 来自图像。它们代表“图片里有什么内容可供关注”。
        # 首先，调整图像特征维度以匹配文本特征维度
        projected_img_map = self.img_proj(image_features_map)  # -> [batch, embed_dim, 64, 64]

        # MultiheadAttention 需要的输入格式是 (batch, sequence_length, embedding_dim)
        # 所以我们需要把 HxW 的特征图拉平成一个序列
        batch_size, embed_dim, h, w = projected_img_map.shape
        # -> [batch, embed_dim, h*w] -> [batch, h*w, embed_dim]
        image_features_seq = projected_img_map.flatten(2).permute(0, 2, 1)

        key = image_features_seq
        value = image_features_seq

        # 3. 执行 Cross-Attention
        # text_features (Q) "关注" image_features_seq (K, V)
        # attn_output 是文本特征关注了图像特定区域后，融合了图像信息的新表征
        # attn_weights 显示了每个文本词汇对图像各区域的注意力权重
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=key,
            value=value
        )

        return attn_output, attn_weights


# --- 3. 实例化并运行模型 ---

# 定义模型参数
EMBED_DIM = 64  # 嵌入维度，需要能被 num_heads 整除
NUM_HEADS = 8  # 注意力头的数量

# 创建模型
model = CrossAttentionModel(vocab_size=vocab_size, embed_dim=EMBED_DIM, num_heads=NUM_HEADS)

# 将输入数据送入模型
fused_features, attention_weights = model(image_tensor, text_tensor)

# --- 4. 检查输出 ---
print("--- 输入维度 ---")
print(f"图像输入 (Image Tensor): {image_tensor.shape}")
print(f"文本输入 (Text Tensor): {text_tensor.shape}")
print("\n--- 模型内部维度 ---")
print(f"文本特征 (Query): {model.text_encoder(text_tensor).shape}")
img_feat_map = model.image_encoder(image_tensor)
proj_img_map = model.img_proj(img_feat_map)
img_feat_seq = proj_img_map.flatten(2).permute(0, 2, 1)
print(f"图像特征 (Key/Value): {img_feat_seq.shape}")
print("\n--- 输出维度 ---")
print(f"融合后的特征 (Fused Features): {fused_features.shape}")
print(f"注意力权重 (Attention Weights): {attention_weights.shape}")
print("\n--- 解释 ---")
print(f"融合后的特征维度 [Batch, SeqLen, Dim] = {fused_features.shape}，")
print("这表示对于文本中的每一个词（共5个），都生成了一个融合了图像信息的 64 维新特征。")
print("这个特征可以用于下游任务，例如预测灵巧手的关节角度。")
print(f"\n注意力权重维度 [Batch, QuerySeqLen, KeySeqLen] = {attention_weights.shape}，")
print(f"这代表文本中的每个词（共{max_seq_len}个）对图像的 {64 * 64} 个区域的注意力分布。")