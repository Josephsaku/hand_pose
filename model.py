# model.py 
import torch
import torch.nn as nn
import torchvision.models as models


# 1. 使用预训练的 ResNet 作为新的图像编码器
class ResNetImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载预训练的 ResNet18
        # 使用推荐的权重参数 weights=models.ResNet18_Weights.DEFAULT
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # ResNet18 在最后一个卷积块后的输出通道数为 512
        self.output_channels = 512

        # 移除最后的平均池化层 (avgpool) 和全连接层 (fc)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # 输入: (B, 3, H, W)
        # 输出: (B, 512, H/32, W/32) 尺寸的特征图
        return self.feature_extractor(x)


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)


# 2. 修改 HandPoseModel 以集成 ResNet
class HandPoseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=8, num_joints=21):
        super().__init__()

        # 使用 ResNet 图像编码器
        self.image_encoder = ResNetImageEncoder()

        self.text_encoder = SimpleTextEncoder(vocab_size, embed_dim)

        # ResNet18 输出 512 个特征通道，所以这里要更新
        img_feature_channels = self.image_encoder.output_channels  # 值为 512

        # 图像投影层，将 512 通道投影到和文本特征一致的维度 (embed_dim)
        self.img_proj = nn.Conv2d(img_feature_channels, embed_dim, kernel_size=1)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        output_dim = num_joints * 3
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, output_dim)
        )

    def forward(self, image, text):
        text_features = self.text_encoder(text)

        # image_features_map 的维度现在是 [B, 512, H', W']
        image_features_map = self.image_encoder(image)

        query = text_features

        # projected_img_map 的维度是 [B, embed_dim, H', W']
        projected_img_map = self.img_proj(image_features_map)

        # 将特征图展平为序列，以输入到注意力模块
        # [B, embed_dim, H', W'] -> [B, embed_dim, H'*W'] -> [B, H'*W', embed_dim]
        image_features_seq = projected_img_map.flatten(2).permute(0, 2, 1)

        key = value = image_features_seq
        attn_output, _ = self.cross_attention(query=query, key=key, value=value)
        fused_features = attn_output.mean(dim=1)
        predicted_coords = self.prediction_head(fused_features)
        return predicted_coords

# import torch
# import torch.nn as nn
#
# class SimpleImageEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.pool(x)
#         return x
#
# class SimpleTextEncoder(nn.Module):
#     def __init__(self, vocab_size, embed_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#     def forward(self, x):
#         return self.embedding(x)
#
# class HandPoseModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim=64, num_heads=8, num_joints=21):
#         super().__init__()
#         self.image_encoder = SimpleImageEncoder()
#         self.text_encoder = SimpleTextEncoder(vocab_size, embed_dim)
#         img_feature_channels = 16
#         self.img_proj = nn.Conv2d(img_feature_channels, embed_dim, kernel_size=1)
#         self.cross_attention = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             batch_first=True
#         )
#         output_dim = num_joints * 3
#         self.prediction_head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.ReLU(),
#             nn.Linear(embed_dim // 2, output_dim)
#         )
#     def forward(self, image, text):
#         text_features = self.text_encoder(text)
#         image_features_map = self.image_encoder(image)
#         query = text_features
#         projected_img_map = self.img_proj(image_features_map)
#         _, embed_dim, _, _ = projected_img_map.shape
#         image_features_seq = projected_img_map.flatten(2).permute(0, 2, 1)
#         key = value = image_features_seq
#         attn_output, _ = self.cross_attention(query=query, key=key, value=value)
#         fused_features = attn_output.mean(dim=1)
#         predicted_coords = self.prediction_head(fused_features)
#         return predicted_coords
