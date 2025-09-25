import torch
from torchvision import transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model import HandPoseModel

#配置参数
MODEL_PATH = 'hand_pose_model.pth'
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

    # 提取 x, y, z 坐标
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # 绘制关节点
    ax.scatter(x, y, z, c='r', marker='o')

    # 绘制骨架
    for start_joint, end_joint in HAND_SKELETON:
        ax.plot([x[start_joint], x[end_joint]],
                [y[start_joint], y[end_joint]],
                [z[start_joint], z[end_joint]], 'b-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Predicted 3D Hand Pose')

    # 调整视角和坐标轴范围，让显示效果更好
    ax.view_init(elev=30., azim=120)

    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # 实例化一个和训练时结构完全相同的模型
    model = HandPoseModel(vocab_size=len(VOCAB), embed_dim=EMBED_DIM, num_heads=NUM_HEADS)

    # 加载训练好的权重
    model.load_state_dict(torch.load(MODEL_PATH))

    # 将模型移动到设备
    model.to(device)

    # 这会关闭 Dropout 等只在训练时使用的层，确保预测结果稳定
    model.eval()

    #准备单张图片和文本
    # 定义和训练时完全一样的图像预处理
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
    # 增加一个维度 (B, C, H, W)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 准备文本输入
    token_ids = [VOCAB.get(word, 0) for word in GLOBAL_INSTRUCTION.split()]
    token_ids += [VOCAB['<pad>']] * (MAX_SEQ_LEN - len(token_ids))
    text_tensor = torch.LongTensor(token_ids)
    # 同样增加一个批次维度 (B, SeqLen)
    text_tensor = text_tensor.unsqueeze(0).to(device)

    #执行预测
    with torch.no_grad():
        predictions = model(image_tensor, text_tensor)

    # 处理并可视化结果
    # 将预测结果从 GPU 移回 CPU，并转换为 NumPy 数组
    predicted_xyz = predictions.cpu().numpy()
    # 模型的输出形状是 (1, 63)，去掉批次维度并重塑为 (21, 3)
    predicted_xyz = predicted_xyz.squeeze().reshape(21, 3)

    print("\n预测完成，正在生成3D姿态可视化图...")
    # 调用可视化函数
    visualize_3d_pose(predicted_xyz)