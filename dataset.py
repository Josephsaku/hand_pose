import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class FreiHANDDataset(Dataset):
    def __init__(self, base_path, split='training', transform=None):
        self.data_dir = os.path.join(base_path, split)
        self.rgb_path = os.path.join(self.data_dir, 'rgb')
        self.transform = transform

        json_filename = ""
        if split == 'training':
            json_filename = 'training_xyz.json'
        elif split == 'validation' or split == 'testing':
            json_filename = 'evaluation_xyz.json'

        xyz_path = os.path.join(self.data_dir, json_filename)
        self.xyz_list = None
        if json_filename and os.path.exists(xyz_path):
            print(f"正在加载标注文件: {xyz_path}")
            with open(xyz_path, 'r') as f:
                self.xyz_list = json.load(f)
        else:
            print(f"警告：在路径 {xyz_path} 中找不到标注文件。")

        all_image_files = sorted(os.listdir(self.rgb_path))
        num_labels = len(self.xyz_list) if self.xyz_list else 0
        self.image_files = all_image_files[:num_labels]

        if self.xyz_list and len(self.image_files) != len(self.xyz_list):
            print(f"严重警告：切片后图片数量({len(self.image_files)})仍与标签数量({len(self.xyz_list)})不匹配")
        else:
            print(f" 图片与标签数量匹配: {len(self.image_files)}")

        #全局指令
        global_instruction = "imitate the hand pose in the image"
        self.vocab = {'<pad>': 0, 'imitate': 1, 'the': 2, 'hand': 3, 'pose': 4, 'in': 5, 'image': 6}
        self.max_seq_len = 10
        token_ids = [self.vocab.get(word, 0) for word in global_instruction.split()]
        token_ids += [self.vocab['<pad>']] * (self.max_seq_len - len(token_ids))
        self.global_text_tensor = torch.LongTensor(token_ids)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.rgb_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.xyz_list:
            label = torch.tensor(self.xyz_list[idx])
            label = label.flatten()
        else:
            label = torch.zeros(63)
        return image, self.global_text_tensor, label
