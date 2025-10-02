import torch
from torchvision import transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model import HandPoseModel

# --- 1. 配置参数更新 ---
# 注释掉旧的模型路径
# MODEL_PATH = 'hand_pose_model.pth'
# 使用新的检查点文件路径
MODEL_PATH = 'training_checkpoint.pth'
DATA_DIR = 'data/FreiHAND/'
TEST_SPLIT = 'testing'

# 模型参数
EMBED_DIM = 64
NUM_HEADS = 8

# 全局文本指令参数
VOCAB = {'<pad>': 0, 'imitate': 1, 'the': 2, 'hand': 3, 'pose': 4, 'in': 5, 'image': 6}
MAX_SEQ_LEN = 10
GLOBAL_INSTRUCTION = "imitate the hand pose in the image"

# 手部骨架连接关系
HAND_SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 4],  # 拇指
    [0, 5], [5, 6], [6, 7], [7, 8],  # 食指
    [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
    [0, 13], [13, 14], [14, 15], [15, 16],  # 无名指
    [0, 17], [17, 18], [18, 19], [19, 20]  # 小指
]


def visualize_3d_pose(xyz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    ax.scatter(x, y, z, c='r', marker='o')

    for start_joint, end_joint in HAND_SKELETON:
        ax.plot([x[start_joint], x[end_joint]],
                [y[start_joint], y[end_joint]],
                [z[start_joint], z[end_joint]], 'b-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Predicted 3D Hand Pose')

    ax.view_init(elev=30., azim=120)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # 实例化一个和训练时结构完全相同的模型
    model = HandPoseModel(vocab_size=len(VOCAB), embed_dim=EMBED_DIM, num_heads=NUM_HEADS)

    # --- 2. 修改模型加载逻辑 ---
    print(f"正在从检查点文件加载模型: {MODEL_PATH}")

    # --- 注释掉旧的加载方式 (假设文件里直接是模型权重) ---
    # model.load_state_dict(torch.load(MODEL_PATH))

    # --- 引入新的加载方式 (从包含状态字典的检查点文件中加载) ---
    # 首先加载包含所有状态的完整检查点字典
    checkpoint = torch.load(MODEL_PATH)
    # 然后，只从字典中提取模型的权重('model_state_dict')来加载到模型中
    model.load_state_dict(checkpoint['model_state_dict'])


    # 将模型移动到设备
    model.to(device)

    # 这会关闭 Dropout 等只在训练时使用的层，确保预测结果稳定
    model.eval()

    # --- 后续的预测和可视化代码保持不变 ---

    #准备单张图片和文本
    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 从测试集中随机选择一张图片
    test_rgb_dir = os.path.join(DATA_DIR, TEST_SPLIT, 'rgb')
    random_image_name = random.choice(os.listdir(test_rgb_dir))
    image_path = os.path.join(test_rgb_dir, random_image_name)
    print(f"\n正在测试图片: {image_path}")

    # 加载并预处理图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = data_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 准备文本输入
    token_ids = [VOCAB.get(word, 0) for word in GLOBAL_INSTRUCTION.split()]
    token_ids += [VOCAB['<pad>']] * (MAX_SEQ_LEN - len(token_ids))
    text_tensor = torch.LongTensor(token_ids)
    text_tensor = text_tensor.unsqueeze(0).to(device)

    #执行预测
    with torch.no_grad():
        predictions = model(image_tensor, text_tensor)

    # 处理并可视化结果
    predicted_xyz = predictions.cpu().numpy()
    predicted_xyz = predicted_xyz.squeeze().reshape(21, 3)

    print("\n预测完成，正在生成3D姿态可视化图...")
    # 调用可视化函数
    visualize_3d_pose(predicted_xyz)