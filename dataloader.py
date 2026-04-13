import torch
from torch.utils.data import Dataset, DataLoader

class SCDM_Dataset(Dataset):
    def __init__(self, pt_file_path):
        """
        专门针对我们在 dataset_splitter.py 里切分并保存好的 .pt 字典而设计的数据加载器。
        它可以无缝读取如 dataset_splits/standard_split/train.pt 里的三个核心矩阵。
        """
        super().__init__()
        data_dict = torch.load(pt_file_path, map_location='cpu')
        
        # 强制转换为 Float32，防止 Float64 前向传播时报类型不匹配或者爆显存
        self.eeg = data_dict['eeg'].float()      # 形状理应为: (N, 30, 4000)
        self.fnirs = data_dict['fnirs'].float()  # 形状理应为: (N, 36, 256)
        self.labels = data_dict['labels'].long() # 形状理应为: (N,)
        
        print(f"Loaded {len(self.labels)} samples from {pt_file_path}.")
        print(f" - EEG Shape: {self.eeg.shape}")
        print(f" - fNIRS Shape: {self.fnirs.shape}")
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.eeg[idx], self.fnirs[idx], self.labels[idx]

def get_dataloaders(train_path, val_path, batch_size=16, num_workers=0):
    """
    工厂函数：一键返回配置好的 train_loader 和 val_loader
    """
    print("\n=> Building Train Dataset...")
    train_dataset = SCDM_Dataset(train_path)
    
    print("\n=> Building Validation Dataset...")
    val_dataset = SCDM_Dataset(val_path)
    
    # 训练集需要打乱，切记设置 drop_last=True，防止最后不满足 Batch 尺寸导致 BatchNorm 崩溃
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, drop_last=True, pin_memory=True)
                              
    # 验证集无需打乱，直接前向传播即可
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, drop_last=False, pin_memory=True)
    
    return train_loader, val_loader
