# # main.py
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from model import HandPoseModel
# from dataset import FreiHANDDataset
#
# # --- 超参数和配置 ---
# LEARNING_RATE = 1e-4
# NUM_EPOCHS = 100
# BATCH_SIZE = 32
# DATA_DIR = 'data/FreiHAND/'
# EMBED_DIM = 64
# NUM_HEADS = 8
#
# # --- 设置运行设备 ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"将使用设备: {device}")
#
# # --- 准备数据 ---
# data_transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# train_dataset = FreiHANDDataset(base_path=DATA_DIR, split='training', transform=data_transform)
# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#
# val_dataset = FreiHANDDataset(base_path=DATA_DIR, split='validation', transform=data_transform)
# val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
# # 实例化模型、损失函数和优化器
# VOCAB_SIZE = len(train_dataset.vocab)
# model = HandPoseModel(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS).to(device)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#
# # 完整的训练与验证循环
# print("\n开始训练 ")
# for epoch in range(NUM_EPOCHS):
#     model.train()
#     running_loss = 0.0
#     for i, (images, texts, labels) in enumerate(train_loader):
#         images, texts, labels = images.to(device), texts.to(device), labels.to(device)
#         optimizer.zero_grad()
#         predictions = model(images, texts)
#         loss = criterion(predictions, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if (i + 1) % 100 == 0:
#             print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
#
#     epoch_train_loss = running_loss / len(train_loader)
#
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for images, texts, labels in val_loader:
#             images, texts, labels = images.to(device), texts.to(device), labels.to(device)
#             predictions = model(images, texts)
#             loss = criterion(predictions, labels)
#             val_loss += loss.item()
#
#     epoch_val_loss = val_loss / len(val_loader)
#
#     print("-" * 30)
#     print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] 完成!')
#     print(f'平均训练损失 (Avg Train Loss): {epoch_train_loss:.4f}')
#     print(f'平均验证损失 (Avg Val Loss):   {epoch_val_loss:.4f}')
#     print("-" * 30)
#
# print("\n--- 训练完成 ---")
# torch.save(model.state_dict(), 'hand_pose_model.pth')
# print("模型已保存到 hand_pose_model.pth")

# main.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from model import HandPoseModel
from dataset import FreiHANDDataset

# --- 超参数和配置 ---
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 32
DATA_DIR = 'data/FreiHAND/'
EMBED_DIM = 64
NUM_HEADS = 8
# --- 检查点文件路径 ---
CHECKPOINT_PATH = 'training_checkpoint.pth'

# --- 设置运行设备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用设备: {device}")

# --- 准备数据 ---
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = FreiHANDDataset(base_path=DATA_DIR, split='training', transform=data_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = FreiHANDDataset(base_path=DATA_DIR, split='validation', transform=data_transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 实例化模型、损失函数和优化器
VOCAB_SIZE = len(train_dataset.vocab)
model = HandPoseModel(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS).to(device)
criterion = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 加载检查点逻辑
start_epoch = 0
best_val_loss = float('inf')

if os.path.exists(CHECKPOINT_PATH):
    print(f"发现已存在的检查点文件: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

    print(f"成功加载检查点。将从 Epoch {start_epoch + 1} 开始继续训练。")
    print(f"已记录的最佳验证损失为: {best_val_loss:.4f}")
else:
    print("未发现检查点文件，将从头开始训练。")

# 完整的训练与验证循环
print(f"\n开始训练，从 Epoch {start_epoch + 1} 到 {NUM_EPOCHS}")
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (images, texts, labels) in enumerate(train_loader):
        images, texts, labels = images.to(device), texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images, texts)
        loss = criterion(predictions, labels)  # 使用 criterion
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, texts, labels in val_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            predictions = model(images, texts)
            loss = criterion(predictions, labels)  # 使用 criterion
            val_loss += loss.item()

    epoch_val_loss = val_loss / len(val_loader)

    print("-" * 30)
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] 完成!')
    print(f'平均训练损失 (Avg Train Loss): {epoch_train_loss:.4f}')
    print(f'平均验证损失 (Avg Val Loss):   {epoch_val_loss:.4f}')

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss

        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }

        torch.save(checkpoint_data, CHECKPOINT_PATH)
        print(f'** 新的最佳模型检查点已保存! 验证损失降至: {best_val_loss:.4f} **')

    print("-" * 30)

print("\n--- 训练完成 ---")
print(f"训练结束。表现最好的模型检查点保存在 {CHECKPOINT_PATH}")