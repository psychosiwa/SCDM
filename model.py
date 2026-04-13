import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Alignment Module (特征统一对齐)
# ==========================================
class Alignment(nn.Module):
    def __init__(self):
        super().__init__()
        # EEG: (B, 30, 4000) -> (B, 32, 256)
        # 用一维卷积配合较大的 stride 和 kernel_size，再加上 AdaptiveAvgPool1d 保证对齐到长度256，通道为32
        self.e_conv = nn.Conv1d(in_channels=30, out_channels=32, kernel_size=15, stride=7, padding=7) 
        self.e_pool = nn.AdaptiveAvgPool1d(256)
        
        # fNIRS: (B, 36, 256) -> (B, 32, 256)
        # 仅做通道维度的线性变换
        self.f_conv = nn.Conv1d(in_channels=36, out_channels=32, kernel_size=1)
        
    def forward(self, e, f):
        e_align = self.e_pool(self.e_conv(e))    # (B, 32, 256)
        f_align = self.f_conv(f)                 # (B, 32, 256)
        return e_align, f_align


# ==========================================
# 2. Time Embedding
# ==========================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.time_embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def get_sinusoidal_embedding(self, timesteps):
        """生成基于正弦/余弦的时间步连续嵌入"""
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # 若 dim 为奇数，可通过 pad 对齐
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t):
        """仅对离散时间步 t 进行特征嵌入 (依据论文 beta_t 仅用于最后公式计算)"""
        t_emb = self.get_sinusoidal_embedding(t)
        t_emb = self.time_embed(t_emb)                   # (B, dim)
        return t_emb


# ==========================================
# 3. SCG (Spatial Cross-modal Generation)
# ==========================================
class SCG(nn.Module):
    def __init__(self, c_q_ch, c_k_ch, d_in, d_out):
        super().__init__()
        # 依据原文: 2-dim 卷积核数为 d_in 作用于 C_q(如Cef), 生成 Q 属于 d_in x 16 x 16
        self.conv_q = nn.Conv2d(c_q_ch, d_in, kernel_size=1)
        # 同样应用到 C_k(如Cfe)，生成 K 属于 d_out x 16 x 16
        self.conv_k = nn.Conv2d(c_k_ch, d_out, kernel_size=1)
        # 最后对齐通道
        self.conv_v = nn.Conv1d(d_out, d_out, kernel_size=1)
        self.d_out = d_out
        self.d_in = d_in

    def forward(self, V, C_q, C_k):
        # V: (B, d_in, L) EEG 表征用作 Value
        # C_q: (B, c_q_ch, 16, 16)
        # C_k: (B, c_k_ch, 16, 16)
        B, d_in_V, L = V.shape
        
        # 1. 产生 Q 和 K，并铺平特征便于注意力计算
        Q = self.conv_q(C_q).view(B, self.d_in, -1)   # (B, d_in, 256)
        K = self.conv_k(C_k).view(B, self.d_out, -1)  # (B, d_out, 256)
        
        # 2. Score = Softmax( Q K^T / sqrt(d_out) )
        # Q: (B, d_in, 256), K^T: (B, 256, d_out) 
        # scores: (B, d_in, d_out)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_out)
        
        # Softmax 在 EEG 通道(dim=1)上归一化，得到每个 EEG 为特定 fNIRS 贡献的分布比重
        attn = F.softmax(scores, dim=1) # (B, d_in, d_out)
        
        # 3. 输出 = Score * V 
        # Score为(B, d_in, d_out)，V为(B, d_in, L)，期望输出d_out个通道特征
        # 数学计算等于 Score^T @ V => (B, d_out, d_in) @ (B, d_in, L) -> (B, d_out, L)
        integrated_V = torch.bmm(attn.transpose(1, 2), V)
        
        # 4. 融合过一维卷积通道映射
        out = self.conv_v(integrated_V) # (B, d_out, L)
        return out


# ==========================================
# 4. MTR (Multi-scale Temporal Representation)
# ==========================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride=1):
        super().__init__()
        # 左侧 padding = (kernel_size - 1) * dilation，保证严格因果关系 (不依赖未来信息)
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        # 仅在左侧进行 padding 填充
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class MTR(nn.Module):
    def __init__(self, in_channels, out_channels, is_downsample=True):
        super().__init__()
        # (1) Causal Dilation Convolution: 3层 1D 卷积 (dilation=1,2,4)
        # 为保留尺度，内部特征提取时 stride 设置为 1，重采样功能专门由 Point-wise Convolution 承担
        self.causal1 = CausalConv1d(in_channels, in_channels, kernel_size=3, dilation=1)
        self.causal2 = CausalConv1d(in_channels, in_channels, kernel_size=3, dilation=2)
        self.causal3 = CausalConv1d(in_channels, in_channels, kernel_size=3, dilation=4)
        
        # (2) Multi-Scale Depth-wise Convolution: 4层独立 1D 深度可分离卷积 (groups=in_channels)
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size=k, padding=k//2, groups=in_channels)
            for k in [3, 5, 7, 9]
        ])
        
        self.is_downsample = is_downsample
        if is_downsample:
            # (3.a) Multi-Scale Point-wise Convolution (下采样): 卷积核1, stride 2
            self.resample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            # (3.b) Up-sample (转置卷积 + 双线性插值配合使用)
            # - 转置分支: ConvTranspose1d 倍增时序维度
            self.up_trans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
            # - 线性分支: 插值扩大2倍 + Conv1D映射通道
            self.up_linear = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            
    def forward(self, x):
        # 1. Causal Dilation Convolution 串联
        x = F.silu(self.causal1(x))
        x = F.silu(self.causal2(x))
        x = F.silu(self.causal3(x))
        
        # 2. Multi-Scale Depth-wise Convolution 并行融合
        dw_outs = [conv(x) for conv in self.dw_convs]
        x = sum(dw_outs)  # 融合多个尺度
        
        # 3. Resampling 重采样
        if self.is_downsample:
            out = self.resample(x)
        else:
            trans_feat = self.up_trans(x)
            # 双线性插值在 1D 时即为 'linear'
            linear_feat = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
            linear_feat = self.up_linear(linear_feat)
            out = trans_feat + linear_feat # 配合使用
            
        return out


# ==========================================
# 5. Representation Block
# ==========================================
class RepresentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, c_eeg_ch=30, c_fnirs_ch=36, t_dim=256, is_downsample=True):
        super().__init__()
        # Main SCG (模态内相关性): 作用于进入 MTR 处理之前
        # 内部映射 d_in -> d_out 皆为 in_channels
        self.main_scg_eeg = SCG(c_q_ch=c_eeg_ch, c_k_ch=c_eeg_ch, d_in=in_channels, d_out=in_channels)
        self.main_scg_fnirs = SCG(c_q_ch=c_fnirs_ch, c_k_ch=c_fnirs_ch, d_in=in_channels, d_out=in_channels)
        
        # Multi-scale Temporal Representation
        self.mtr_eeg = MTR(in_channels, out_channels, is_downsample)
        self.mtr_fnirs = MTR(in_channels, out_channels, is_downsample)
        
        # Translation SCG (跨模态相关性): 把 EEG 表征隐射给 fNIRS
        # C_ef 为 Q 提供源 (c_eeg_ch)，C_fe 为 K 提供源 (c_fnirs_ch)
        self.trans_scg = SCG(c_q_ch=c_eeg_ch, c_k_ch=c_fnirs_ch, d_in=out_channels, d_out=out_channels)
        
        # 条件注入线性映射
        self.eeg_time_proj = nn.Linear(t_dim, in_channels)
        self.fnirs_time_proj = nn.Linear(t_dim, in_channels)
        
    def forward(self, eeg_feat, fnirs_feat, Ce, Cf, Cef, Cfe, t_emb):
        # 提取当前特征通道长度供后续条件注入
        curr_channels = eeg_feat.shape[1]
        
        # [控制信号注入] 将 Time Embedding 注入到后续所有 Block 中
        # (B, t_dim) -> (B, in_channels, 1) 加入 broadcast
        e_emb_proj = self.eeg_time_proj(t_emb).unsqueeze(-1)
        f_emb_proj = self.fnirs_time_proj(t_emb).unsqueeze(-1)
        eeg_feat = eeg_feat + e_emb_proj
        fnirs_feat = fnirs_feat + f_emb_proj
        
        # [Main SCG] 模态内聚集
        eeg_feat = self.main_scg_eeg(eeg_feat, Ce, Ce)
        fnirs_feat = self.main_scg_fnirs(fnirs_feat, Cf, Cf)
        
        # [MTR 模块] 提取并下采样/上采样 (in_channels -> out_channels)
        eeg_feat = self.mtr_eeg(eeg_feat)
        fnirs_feat = self.mtr_fnirs(fnirs_feat)
        
        # [Translation SCG] 每经过一个 Block 的 EEG_feat，进行跨模态翻译
        # 数据特征V取 EEG_feat, 提供跨模态的 Query 对应 Cef，Key 对应 Cfe
        trans_out = self.trans_scg(eeg_feat, Cef, Cfe)
        
        # [Residual Connection] 跨模态翻译结果与当前的 fNIRS_feat 构成残差相加
        fnirs_feat = fnirs_feat + trans_out
        
        return eeg_feat, fnirs_feat


# ==========================================
# 6. D_theta: 核心扩散去噪网络
# ==========================================
class Denoising_Net(nn.Module):
    def __init__(self, t_dim=256):
        super().__init__()
        self.align = Alignment()
        self.time_embed_module = TimeEmbedding(dim=t_dim)
        
        # U-Net 形态的 6 个 Block (3 Down-sample + 3 Up-sample)
        # 前三个负责通道放大并时序二分之一特征下采样
        self.down_blocks = nn.ModuleList([
            RepresentationBlock(32, 64, c_eeg_ch=30, c_fnirs_ch=36, t_dim=t_dim, is_downsample=True),
            RepresentationBlock(64, 128, c_eeg_ch=30, c_fnirs_ch=36, t_dim=t_dim, is_downsample=True),
            RepresentationBlock(128, 256, c_eeg_ch=30, c_fnirs_ch=36, t_dim=t_dim, is_downsample=True)
        ])
        
        # 后三个负责通道缩小并时序二倍特征上采样
        self.up_blocks = nn.ModuleList([
            RepresentationBlock(256, 128, c_eeg_ch=30, c_fnirs_ch=36, t_dim=t_dim, is_downsample=False),
            RepresentationBlock(128, 64, c_eeg_ch=30, c_fnirs_ch=36, t_dim=t_dim, is_downsample=False),
            RepresentationBlock(64, 32, c_eeg_ch=30, c_fnirs_ch=36, t_dim=t_dim, is_downsample=False)
        ])
        
        # 6个 Block 产生的 6 个张量 Channel 累计之和: 64+128+256+128+64+32 = 672
        # Final Conv 用于融合最终 Skip connections 并恢复与原 fNIRS 输入一致的 36 通道
        self.final_conv = nn.Conv1d(672, 36, kernel_size=1)

    def forward(self, e_t, f_t, t, beta_t, Ce, Cf, Cef, Cfe, alpha_t=None, alpha_bar_t=None):
        # 1. 模态对齐转为 (B, 32, 256)
        eeg_feat, fnirs_feat = self.align(e_t, f_t)
        
        # 2. 生成纯净的时间控制特征 (beta_t 退居幕后仅用于去噪步计算)
        t_emb = self.time_embed_module(t)
        
        skip_connections = []
        
        # 3. 前 3 个 Block: 下采样阶段
        for block in self.down_blocks:
            eeg_feat, fnirs_feat = block(eeg_feat, fnirs_feat, Ce, Cf, Cef, Cfe, t_emb)
            # 存入 skip_connections 列表
            skip_connections.append(fnirs_feat)
            
        # 4. 后 3 个 Block: 上采样阶段
        for block in self.up_blocks:
            eeg_feat, fnirs_feat = block(eeg_feat, fnirs_feat, Ce, Cf, Cef, Cfe, t_emb)
            # 存入 skip_connections 列表
            skip_connections.append(fnirs_feat)
            
        # 5. 最后将 skip_connections 中的 6 个张量进行 Add & Concatenate，恢复原始形状
        unified_feats = []
        for feat in skip_connections:
            # Add/Interpolate 保证时序长全部对齐为 256
            if feat.shape[-1] != 256:
                feat = F.interpolate(feat, size=256, mode='linear', align_corners=False)
            unified_feats.append(feat)
            
        # Concatenate 所有特征
        concat_feat = torch.cat(unified_feats, dim=1) # (B, 672, 256)
        
        # 6. 恢复成 (B, 36, 256) 形状，得到特征表示(或者物理意义上的预测噪声 epsilon_theta)
        pred_noise = self.final_conv(concat_feat)
        
        # 7. 根据要求，在这步网络的最末端直接执行 Denoising 去噪公式。输出预测的 f!
        if alpha_t is not None and alpha_bar_t is not None:
            f_hat_t_minus_1 = (1.0 / torch.sqrt(alpha_t)) * (
                f_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
            )
            return f_hat_t_minus_1
        else:
            # 降级备用分支，倘若未传入对应的 alpha 时
            return pred_noise


# ==========================================
# 7. 训练与推理统筹调度类 
# ==========================================
class SCDM_Trainer:
    def __init__(self, model: Denoising_Net, num_timesteps=1000, device='cuda', lr=1e-4, beta_schedule=None):
        self.model = model.to(device)
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 设定调度器参数 \beta_{1} 到 \beta_{T}
        # 根据原文, T 和 beta_t 系列是由最小化 Wasserstein 距离在模型训练前 (prior to training) 确定的
        if beta_schedule is not None:
            self.beta = beta_schedule.to(device)
            self.num_timesteps = len(self.beta)
        else:
            # 备用：默认的线性生成策略
            self.beta = torch.linspace(1e-4, 0.02, self.num_timesteps).to(device)
            
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, e0, f0, Ce, Cf, Cef, Cfe):
        """
        [Algorithm 1: Training Phase of the SCDM]
        严格执行扩散与网络去噪前向，并反向传播基于目标步的拟合误差。
        """
        self.model.train()
        B = e0.shape[0]
        
        # 1. 扩散过程 (Diffusion Process) 设定 t
        t = torch.randint(1, self.num_timesteps + 1, (B,), device=self.device)
        beta_t = self.beta[t - 1]
        alpha_t = self.alpha[t - 1].view(-1, 1, 1)
        alpha_bar_t = self.alpha_bar[t - 1].view(-1, 1, 1)
        
        # 采样高斯噪声 \epsilon_t ~ N(0, I)
        epsilon_e = torch.randn_like(e0)
        epsilon_f = torch.randn_like(f0)
        
        # 通过直接闭合公式生成带有扰动噪声的中间态 e_t与 f_t (数学等价于多步马尔可夫加噪)
        e_t = torch.sqrt(alpha_bar_t) * e0 + torch.sqrt(1 - alpha_bar_t) * epsilon_e
        f_t = torch.sqrt(alpha_bar_t) * f0 + torch.sqrt(1 - alpha_bar_t) * epsilon_f
        
        # 提取或者推算真实标签 f_ (这里取对应步骤的前向理想推导值作为目标或者如果您的应用设定f_t直接做监督)
        alpha_bar_t_minus_1 = torch.where(t > 1, self.alpha_bar[t - 2], torch.ones_like(self.alpha_bar[0]))
        alpha_bar_t_minus_1 = alpha_bar_t_minus_1.view(-1, 1, 1)
        target_f_t_minus_1 = torch.sqrt(alpha_bar_t_minus_1) * f0 + torch.sqrt(1 - alpha_bar_t_minus_1) * epsilon_f
        
        # 2. 网络最末层直接执行 Denoising 去噪公式。输出预测的 f_hat_t
        f_hat_t_minus_1 = self.model(e_t, f_t, t, beta_t, Ce, Cf, Cef, Cfe, alpha_t, alpha_bar_t)
        
        # 3. 反向传播 (Backpropagation) 基于 f_ 和预测出的 f_hat 的直接差距 MSE
        self.optimizer.zero_grad()
        loss = self.criterion(f_hat_t_minus_1, target_f_t_minus_1)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def inference(self, e_t_seq, Ce, Cf, Cef, Cfe):
        """
        [Algorithm 2: Inference Phase of the SCDM]
        基于输入的测试集跨模态对应扩散条件 e_t，逆向预测生成真实 \hat{f}_0。
        注意此处入参采用函数外事先处理过的对应步长 EEG 信号表征。
        """
        self.model.eval()
        # 确定测试样本形态，假定取自测试集并具备相同空间长
        B = e_t_seq.shape[0]
        
        # 初始化采样高斯噪声 \hat{f}_T ~ N(0, I)
        # 模型输出需满足 (B, 36, 256) 的 fNIRS 形态
        f_hat_t = torch.randn(B, 36, 256, device=self.device)
        
        # 对于 t = T 倒序到 1
        for i in reversed(range(1, self.num_timesteps + 1)):
            t_tensor = torch.full((B,), i, device=self.device, dtype=torch.long)
            beta_t = self.beta[t_tensor - 1]
            alpha_t = self.alpha[t_tensor - 1].view(-1, 1, 1)
            alpha_bar_t = self.alpha_bar[t_tensor - 1].view(-1, 1, 1)
            
            e_t_input = e_t_seq 
            # 内部网络最后一步进行了 Denoising，直接输出当前阶段的去噪结果 \hat{f}
            f_hat_t_minus_1 = self.model(e_t_input, f_hat_t, t_tensor, beta_t, Ce, Cf, Cef, Cfe, alpha_t, alpha_bar_t)
            
            # 由于网络输出为无随机项的确定性推导去噪状态，此时需额外把方差项补充回去以维持马尔可夫采样链
            if i > 1:
                z = torch.randn_like(f_hat_t)
                sigma_t = torch.sqrt(beta_t).view(-1, 1, 1)
                f_hat_t_minus_1 = f_hat_t_minus_1 + sigma_t * z
                
            f_hat_t = f_hat_t_minus_1
            
        # 最终得到的为不带噪声的预测信号
        return f_hat_t
