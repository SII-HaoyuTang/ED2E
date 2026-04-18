# EDBench 体素化基线训练说明

## 目标

在 EDBench 的电子密度数据上训练一个 **3D 体素卷积回归网络**，网络结构参照原始 `ELFNet` 3D DenseNet 基线。

当前实现位于：

```text
benchmark/voxel/
├── data/energy_dataset.py
├── models/voxel_densenet.py
├── preprocess_voxel_cache.py
└── train_energy.py

benchmark/outputs/
├── checkpoints/
│   ├── benchmark/
│   └── train/
└── wandb/
    ├── benchmark/
    └── train/
```

---

## 与原始 ELFNet 的对应关系

本仓库的对应实现：

| 参考项目 | 当前实现 |
|---|---|
| `training/densenet_pl.py` | `benchmark/voxel/models/voxel_densenet.py` |
| `training/train_pl.py` | `benchmark/voxel/train_energy.py` |
| `training/utils.py` | `benchmark/voxel/data/energy_dataset.py` |

差异：

- 保留了 `ELFNet` 的 3D DenseNet 主体结构。
- 去掉了旧版 PyTorch Lightning 依赖，改为普通 PyTorch 训练循环。
- 输入不再是 Multiwfn 生成的体素性质网格，而是 **EDBench 的电子密度点云体素化结果**。
- 输出不是单一标量，而是 **6 维能量回归目标**。

---

## 数据输入

体素化训练使用两个原始文件：

```text
data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl
data/ed_energy_5w/raw/ed_energy_5w.csv
```

其中：

- `mol_EDthresh0.05_data.pkl` 提供电子密度点云
- `ed_energy_5w.csv` 提供 6 个能量标签和 `scaffold_split` / `random_split`

标签顺序：

```text
E1_Final
E2_NucRepul
E3_OneElec
E4_TwoElec
E5_XC
E6_Total
```

默认使用 `scaffold_split`。

---

## 体素化方式

### 输入通道

当前支持的体素通道：

- `density`：电子密度点落入目标体素后的密度累加
- `atom_occupancy`：原子占据计数
- `atom_z`：原子序数累加

默认只使用：

```text
density
```

### 体素化步骤

对每个分子：

1. 用电子密度加权质心作为体素立方体中心
2. 将坐标裁剪/映射到固定边长立方体 `cube_size_bohr`
3. 按 `grid_length × grid_length × grid_length` 离散化
4. 对应通道做 `scatter-add`
5. 可选做 Gaussian smoothing（`gaussian_sigma > 0` 时）

默认超参数：

- `grid_length = 14`
- `cube_size_bohr = 32.0`
- `channels = density`
- `gaussian_sigma = 0.0`

---

## 预处理：生成 voxel cache

先将原始点云离线转换为单分子 voxel cache：

```bash
python -m benchmark.voxel.preprocess_voxel_cache \
    --pkl-path data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache-dir data/ed_energy_5w/cache_voxel \
    --grid-length 14 \
    --cube-size-bohr 32.0 \
    --channels density \
    --workers 8
```

缓存文件命名形式：

```text
{mol_id}_g14_c32p00_s0p00_density.pt
```

同时会生成一份元数据文件：

```text
meta_g14_c32p00_s0p00_density.json
```

现在离线体素化会显示 `tqdm` 进度条，并使用单进程内的多线程并行处理单分子缓存，避免多进程复制原始 9GB `pkl`。

如需冒烟测试，可限制每个 split 的样本数：

```bash
python -m benchmark.voxel.preprocess_voxel_cache \
    --max-samples-per-split 64
```

如果你直接运行训练脚本且 cache 还不存在，`--max-train-samples`、`--max-val-samples`、`--max-test-samples` 也会约束自动补建的 cache 范围，不会误触发整库体素化。

---

## 训练

### 冒烟测试

```bash
python -m benchmark.voxel.train_energy \
    --pkl-path data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache-dir data/ed_energy_5w/cache_voxel \
    --run-kind benchmark \
    --run-name voxel_smoke \
    --prepare-cache \
    --prepare-cache-workers 4 \
    --grid-length 14 \
    --cube-size-bohr 32.0 \
    --channels density \
    --batch-size 8 \
    --epochs 2 \
    --max-train-samples 128 \
    --max-val-samples 64 \
    --max-test-samples 64 \
    --device cpu
```

### 正式训练

```bash
python -m benchmark.voxel.train_energy \
    --pkl-path data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache-dir data/ed_energy_5w/cache_voxel \
    --run-kind train \
    --run-name voxel_full_density \
    --grid-length 14 \
    --cube-size-bohr 32.0 \
    --channels density \
    --batch-size 32 \
    --epochs 100 \
    --lr 1e-3 \
    --weight-decay 5e-2 \
    --save-every 10 \
    --wandb \
    --wandb-project edbench-voxel \
    --wandb-group voxel \
    --wandb-tags density,grid14 \
    --device cuda
```

训练脚本会：

- 自动读取 train split 目标均值和方差
- 在训练时对 6 个目标做标准化
- 将 checkpoint 保存到 `benchmark/outputs/checkpoints/{benchmark|train}/{run_name}/`
- 保存 `best.pt`、`last.pt` 和周期性 `epoch_XXXX.pt`
- 将 W&B 运行记录写入 `benchmark/outputs/wandb/{benchmark|train}/`
- 输出 `config.json` 和 `metrics.json`

如需启用 W&B，请先在对应训练环境中安装：

```bash
pip install wandb
```

---

## 关键训练参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--grid-length` | `14` | 体素网格边长 |
| `--cube-size-bohr` | `32.0` | 体素立方体实际边长（Bohr） |
| `--channels` | `density` | 输入通道，逗号分隔 |
| `--gaussian-sigma` | `0.0` | 体素平滑强度 |
| `--batch-size` | `32` | batch size |
| `--epochs` | `100` | 训练轮数 |
| `--run-kind` | `benchmark` | 输出归类到 `benchmark` 或 `train` |
| `--run-name` | 自动生成 | 本次实验名，用于目录名与 W&B run name |
| `--prepare-cache-workers` | `4` | 离线体素化线程数 |
| `--save-every` | `10` | 额外周期 checkpoint 保存间隔 |
| `--dense1` | `16` | 第一 dense block 深度 |
| `--dense2` | `16` | 第二 dense block 深度 |
| `--growth-rate` | `12` | DenseNet growth rate |
| `--num-init-features` | `64` | 初始通道数 |
| `--split-col` | `scaffold_split` | 数据划分列 |

---

## 模型结构

当前网络是一个 3D DenseNet 回归器：

```text
Input:  (B, C, G, G, G)
  ↓
Conv3d
  ↓
DenseBlock 1
  ↓
Transition
  ↓
DenseBlock 2
  ↓
BN + ReLU
  ↓
Global Avg Pool
  ↓
Linear → 6
```

输出为 6 维能量预测。

---

## 实现注意点

1. `mol_EDthresh0.05_data.pkl` 是单个大 pickle。预处理现在保持单进程，并通过线程并行单分子体素化，以避免多进程复制这份大对象。
2. 训练阶段不再依赖原始 9GB PKL，只读取单分子 `.pt` cache。
3. 当前默认是单通道 `density` 体素输入。如果你想更贴近“电子密度 + 原子结构”混合输入，可以尝试 `--channels density,atom_occupancy,atom_z`。
4. 如果启用 `--wandb`，训练会记录配置、学习率、训练损失、验证集各目标 MAE/RMSE、测试集最终指标以及最佳 checkpoint 路径。
5. 这是一个 **voxel CNN 基线**。它的作用主要是与 point-based 方法做公平对比，而不是替代当前 ED2E 主干。
6. 这条 benchmark 线现在完全位于 `benchmark/voxel` 下，不再依赖 `ed2e/` 或仓库根目录 `scripts/` 中的 voxel 相关实现。

---

## 实验结果

### 训练配置

| 项目 | 值 |
|------|-----|
| 数据集 | EDBench 47k 子集（scaffold_split） |
| Train / Val / Test | 38,388 / 4,799 / 4,799 |
| 体素参数 | 14³，32 Bohr，单通道 density |
| 模型参数量 | 835,910 |
| Epochs | 400 |
| Batch size | 32 |
| 设备 | RTX 4090 |
| 训练时长 | ~5.2 小时（18,564 秒） |
| **最优 Epoch** | **322**（Val Mean MAE = 39.18 Hartree） |

### 测试集指标（Epoch 322 最优权重）

| 能量目标 | MAE (Hartree) | RMSE (Hartree) |
|---------|--------------|----------------|
| E1_Final | 39.06 | 95.94 |
| E2_NucRepul | 24.42 | 32.50 |
| E3_OneElec | 95.23 | 176.04 |
| E4_TwoElec | 38.63 | 64.22 |
| E5_XC | **2.06** | 3.54 |
| E6_Total | 39.06 | 95.94 |
| **Mean MAE** | **39.74** | — |

### 与 EDBench 论文基准对比

EDBench 论文（NeurIPS 2025）使用完整 3.3M 数据集训练 X-3D 点云模型，作为参考：

| 能量目标 | 本项目 Voxel MAE | X-3D MAE（论文 3.3M） |
|---------|----------------|----------------------|
| E1_Final | 39.06 | 190.77 |
| E2_NucRepul | 24.42 | 109.21 |
| E3_OneElec | 95.23 | 369.88 |
| E4_TwoElec | 38.63 | 150.05 |
| E5_XC | **2.06** | **8.13** |
| E6_Total | 39.06 | 190.77 |

> **注意**：两者使用不同模型（Voxel DenseNet vs X-3D）且训练数据规模差异极大（47k vs 3.3M），不构成公平对比。相对顺序（E5 最易预测，E3 最难）在两者间一致，符合预期。

### 结果分析

训练结束后可运行以下命令生成分析图：

```bash
python -m benchmark.voxel.analyze_results \
    --run-dir  benchmark/outputs/checkpoints/train/voxel_full_density \
    --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache-dir data/ed_energy_5w/cache_voxel \
    --output-dir benchmark/outputs/analysis/voxel \
    --device cpu
```

#### 输出文件

| 文件 | Phase | 内容 |
|------|-------|------|
| `voxel_training_curves.png` | A | train_loss（log）+ val MAE + LR vs epoch；灰色虚线标注 best epoch |
| `voxel_scatter.png` | C | 全 split 叠加散点图（pred vs true，2×3） |
| `voxel_scatter_{train/valid/test}.png` | C | 各 split 单独散点图 |
| `voxel_residual_hist.png` | D1 | test split 残差直方图 + KDE（2×3，每目标一格） |
| `voxel_error_vs_density.png` | D2 | test split 绝对误差 vs 分子体素平均密度（Spearman ρ 注释） |
| `voxel_error_vs_size.png` | D3 | test split 绝对误差 vs 非零体素占比（分子体积代理） |
| `voxel_error_correlation.png` | D4 | 6 个目标绝对误差的 6×6 Pearson 相关热图 |
| `voxel_cdf.png` | E | test split 累积误差分布（CDF），6 条曲线 |
| `analysis_summary.json` | F | per-split per-target MAE / RMSE / Pearson r / Spearman ρ |

#### 各 Phase 分析说明

- **Phase A**：训练过程监控。双 y 轴：左 train_loss（对数刻度），右 val_mean_MAE；灰色虚线标注 best epoch。
- **Phase C**：预测质量核心图。散点越靠近 `y=x` 对角线越好；注释框显示 MAE / RMSE / Pearson r / Spearman ρ（四项指标）。
- **Phase D1**：偏差（bias）分析。残差均值接近 0 表示无系统性偏移；KDE 曲线反映误差分布形态。
- **Phase D2**：误差与体素密度相关性。Spearman ρ 显著正值说明高密度区域预测更难。x 轴为每分子所有体素密度的均值。
- **Phase D3**：误差与分子体积相关性。x 轴为非零体素（> 0.01 a.u.）占总体素数的比例，是分子实际占据空间的代理指标。
- **Phase D4**：各目标误差相关性热图。E1/E6 误差理论上应 100% 相关（两者物理等价）。
- **Phase E**：CDF 曲线。读取 P(|error| < x) 在各 x 处的值，直观比较不同能量目标的误差尺度。

#### 仅生成训练曲线（跳过推理）

```bash
python -m benchmark.voxel.analyze_results \
    --run-dir  benchmark/outputs/checkpoints/train/voxel_full_density \
    --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache-dir data/ed_energy_5w/cache_voxel \
    --skip-inference
```
