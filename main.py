# main.py (最终训练版本)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import HandPoseModel
from dataset import FreiHANDDataset

# --- 超参数和配置 ---
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 32
DATA_DIR = 'data/FreiHAND/'
EMBED_DIM = 64
NUM_HEADS = 8

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
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 完整的训练与验证循环
print("\n开始训练 ")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (images, texts, labels) in enumerate(train_loader):
        images, texts, labels = images.to(device), texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images, texts)
        loss = criterion(predictions, labels)
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
            loss = criterion(predictions, labels)
            val_loss += loss.item()

    epoch_val_loss = val_loss / len(val_loader)

    print("-" * 30)
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] 完成!')
    print(f'平均训练损失 (Avg Train Loss): {epoch_train_loss:.4f}')
    print(f'平均验证损失 (Avg Val Loss):   {epoch_val_loss:.4f}')
    print("-" * 30)

print("\n--- 训练完成 ---")
torch.save(model.state_dict(), 'hand_pose_model.pth')
print("模型已保存到 hand_pose_model.pth")