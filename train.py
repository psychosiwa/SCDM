import torch
import torch.optim as optim
from dataloader import get_dataloaders
from model import SCDM_Trainer
import os

def train_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n===========================================")
    print(f"🚀 INITIALIZING SCDM TRAINING ON: {device}")
    print(f"===========================================")
    
    # ==========================================
    # 1. 加载文件路径配置 (可根据 10折CV 或者 standard 切换)
    # ==========================================
    train_path = 'dataset_splits/standard_split/train.pt'
    val_path = 'dataset_splits/standard_split/val.pt'
    priors_path = 'global_spatial_priors.pt'
    
    if not os.path.exists(train_path) or not os.path.exists(priors_path):
        raise FileNotFoundError("Can't find prepared dataset or spatial priors! Please run dataset_splitter.py and spatial_correlation.py first.")
        
    # ==========================================
    # 2. 从硬盘直接提取先验的 2D 注意力地形矩阵 (只需一次！避免OOM！)
    # ==========================================
    print("\n[1/3] Loading static spatial priors mapping...")
    priors = torch.load(priors_path, map_location=device)
    Ce = priors['Ce'].to(device)
    Cf = priors['Cf'].to(device)
    Cef = priors['Cef'].to(device)
    Cfe = priors['Cfe'].to(device)
    print(f"      -> Injected Geographical Constants: Ce {Ce.shape} / Cef {Cef.shape}")

    # ==========================================
    # 3. 构建 DataLoader 及扩散模型骨架
    # ==========================================
    print("\n[2/3] Constructing Dataloaders and DDPM Network...")
    train_loader, val_loader = get_dataloaders(train_path, val_path, batch_size=16)
    
    model = SCDM_Trainer(
        in_channels_eeg=30, 
        in_channels_fnirs=36, 
        latent_dim=32, 
        scg_dim=64, 
        num_timesteps=1000
    ).to(device)
    
    # 扩散模型标准的衰减优化器配合 AdamW
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    # ==========================================
    # 4. 主循环训练 (Main DDPM Execution Loop)
    # ==========================================
    print("\n[3/3] Commencing Training Epochs...")
    num_epochs = 200
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (eeg, fnirs, labels) in enumerate(train_loader):
            eeg = eeg.to(device)
            fnirs = fnirs.to(device)
            
            optimizer.zero_grad()
            
            # SCDM_Trainer 内置了扩散公式和 Loss 求解，极其干净优雅！
            loss = model(eeg, fnirs, Ce, Cf, Cef, Cfe)
            
            loss.backward()
            # 梯度裁剪：防止在扩散初期加噪网络拟合过度导致的梯度破裂
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        scheduler.step()
        
        # ==========================================
        # 验证评估环节 (仅评估扩散网络误差，不采样子生成)
        # ==========================================
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for eeg, fnirs, labels in val_loader:
                eeg = eeg.to(device)
                fnirs = fnirs.to(device)
                loss = model(eeg, fnirs, Ce, Cf, Cef, Cfe)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch [ {epoch:03d} / {num_epochs} ] | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_scdm_model.pth')
            print(f"  ➜ [CheckPoint] Best Model weights updated (Targeting Val MSE: {best_val_loss:.6f})")

if __name__ == '__main__':
    train_pipeline()
