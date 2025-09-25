import torch
import torch.nn as nn

class SimpleImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        return x

class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    def forward(self, x):
        return self.embedding(x)

class HandPoseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=8, num_joints=21):
        super().__init__()
        self.image_encoder = SimpleImageEncoder()
        self.text_encoder = SimpleTextEncoder(vocab_size, embed_dim)
        img_feature_channels = 16
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
        image_features_map = self.image_encoder(image)
        query = text_features
        projected_img_map = self.img_proj(image_features_map)
        _, embed_dim, _, _ = projected_img_map.shape
        image_features_seq = projected_img_map.flatten(2).permute(0, 2, 1)
        key = value = image_features_seq
        attn_output, _ = self.cross_attention(query=query, key=key, value=value)
        fused_features = attn_output.mean(dim=1)
        predicted_coords = self.prediction_head(fused_features)
        return predicted_coords