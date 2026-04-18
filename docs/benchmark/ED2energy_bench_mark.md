# EDBench ED5-EC 能量预测复现指南

> 目标：从电子密度点云预测 6 种 DFT 能量分量，复现 EDBench 论文（NeurIPS 2025）的 X-3D 基准结果。

---

## 任务说明

**ED5-EC**（Electron Density → 5 Energy Components）是 EDBench 基准测试的核心回归任务之一。

根据 Hohenberg-Kohn 定理，分子的全部基态性质（包括总能量）由其电子密度唯一决定。ED5-EC 任务直接验证这一理论：**仅凭电子密度的三维点云，能否准确预测 DFT 计算的 6 种能量分量？**

### 预测目标（6 种能量，单位 Hartree）

| 索引 | 名称 | 含义 | 典型量级 |
|------|------|------|----------|
| E1 | @DF-RKS Final Energy | DFT 收敛后的最终总能量 | −800 ~ −900 |
| E2 | Nuclear Repulsion Energy | 原子核间排斥能 | +600 ~ +1200 |
| E3 | One-Electron Energy | 动能 + 电子-核吸引能 | −2000 ~ −3500 |
| E4 | Two-Electron Energy | 电子-电子排斥能 | +800 ~ +1600 |
| E5 | DFT Exchange-Correlation Energy | 交换相关能（DFT 核心校正） | −70 ~ −90 |
| E6 | Total Energy | 各分量之和（= E1） | −800 ~ −900 |

---

## 数据

### 来源文件

| 文件 | 大小 | 内容 |
|------|------|------|
| `data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl` | ~9 GB | 47,986 个分子的电子密度点云（已过滤 < 0.05 a.u.） |
| `data/ed_energy_5w/raw/ed_energy_5w.csv` | ~6 MB | 分子 ID、SMILES、6 种能量标签、划分列 |
| `data/ed_energy_5w/raw/readme.md` | — | 数据说明 |

### CSV 列说明

```
index            分子 ID（字符串，与 PKL 键一致）
smiles           原始 SMILES
canonical_smiles 规范化 SMILES
mol_cluster      分子簇 ID（用于 scaffold split）
energy_cluster   能量簇 ID
label            6 个能量值，空格分隔（Hartree）
scaffold_split   train / valid / test（基于分子骨架划分）
random_split     train / valid / test（随机划分，仅供参考）
```

### 数据处理流程

```
PKL 文件
  ├── mol_id → electronic_density.coords  (M, 3)  Bohr
  └── mol_id → electronic_density.density (M,)    a.u.
         ↓
  拼接 [x, y, z, density] → (M, 4)
         ↓
  FPS 采样（最远点采样）→ (2048, 4)          ← 关键！与 EDBench 原始处理一致
         ↓
  缓存为 .pt 文件（{mol_id}_fps2048.pt）
```

> **为什么用 FPS 而非 K-Means？**
> 能量预测任务不需要密度加权的聚类代表点；FPS 保留全局几何结构，与 EDBench 原始实现一致。

> **缓存策略**
> `benchmark/point_cloude/data/energy_dataset.py` 现在只在缓存缺失时惰性加载原始 `pkl`。
> 如果发现缓存未建全，训练脚本会自动将 `num_workers` 收紧到 `0`，避免多个 worker 同时复制 9 GB 原始数据。
>
> **目标标准化**
> `benchmark/point_cloude/train_energy.py` 会基于 `train` split 统计 6 个能量目标的均值和标准差，
> 训练时先做 z-score，再在标准化空间里计算 `MSELoss`。
> 验证 / 测试阶段会将预测反标准化回原始 Hartree 单位后再计算 MAE / RMSE / 相关系数。
> 因此日志中的 `train_loss` 现在通常是 `O(1)`，而不是原先直接对 Hartree 值做 MSE 时的百万量级。

---

## 模型架构：PointMetaBase-S-X3D

### 架构总览

```
输入: (B, 2048, 4)   ← [x, y, z, density]
      ↓
  [Stage 1]  stride=1, width=32,  2048→2048 点
      ↓
  [Stage 2]  stride=2, width=64,  2048→1024 点
      ↓
  [Stage 3]  stride=2, width=128, 1024→512 点  ← X-3D 显式结构特征
      ↓
  [Stage 4]  stride=2, width=256, 512→256 点   ← X-3D 显式结构特征
      ↓
  [Stage 5]  stride=2, width=256, 256→128 点
      ↓
  [Stage 6]  Global SA (max+mean pool) → (B, 256)
      ↓
  [MLP Head] 256 → 512 → 256 → 6
输出: (B, 6)   训练时对应标准化后的 6 维目标；评估时反标准化回 Hartree
```

### 核心组件

#### 1. Set Abstraction（局部聚合）

每个阶段的特征提取：

```
FPS 采样中心点 M 个
  ↓ Ball Query: 半径 r，最多 K=32 个邻居
  ↓ feature_type = "dp_fj": 拼接 [Δxyz | feature_j]
  ↓ LocalAgg MLP
  ↓ Max Pooling over neighbors
  ↓ InvResMLP block（残差精炼）
```

**半径序列**（各阶段倍增）：`0.15 → 0.225 → 0.338 → 0.506 → 0.759 → 1.139` Bohr

#### 2. X-3D 显式结构编码（第 3、4 阶段）

在局部聚合之前，对每个中心点的 K 个邻居提取几何描述符：

**PCA 几何特征（9 维）**：
- 线性度（linearity）= (λ₁ − λ₂) / λ₁
- 平面度（planarity）= (λ₂ − λ₃) / λ₁
- 散射度（scattering）= λ₃ / λ₁
- 全向性（omnivariance）= (λ₁λ₂λ₃)^(1/3)
- 各向异性（anisotropy）= (λ₁ − λ₃) / λ₁
- 三个特征值 λ₁ ≥ λ₂ ≥ λ₃
- 主方向的垂直分量

**PointHop 特征（24 维）**：
- 按坐标轴正负将邻居分为 8 个象限
- 每个象限内邻居的平均相对坐标（3 维）
- 8 × 3 = 24 维

**NeighborContext（动态权重）**：
- 用 33 维几何特征生成 attention 权重（`weight_gen`：MLP + Sigmoid）
- 对邻居特征加权后 Max Pooling
- 通过 Conv1d pipeline 输出

#### 3. InvResMLP（逆残差 MLP）

```
x → Linear(C→4C) → BN → GELU → Linear(4C→C) → BN → + x
```

#### 4. RegressionHead

```
Linear(global_dim, 512) → BN → ReLU → Dropout(0.5)
Linear(512, 256)        → BN → ReLU → Dropout(0.5)
Linear(256, 6)
```

---

## 与 EDBench 原始实现的对应关系

| EDBench 原始文件 | 本项目对应文件 |
|----------------|--------------|
| `openpoints/models/backbone/pointmetabase_X3D.py` | `benchmark/point_cloude/models/backbone/pointmetabase_x3d.py` |
| `openpoints/models/backbone/X_3D_utils/explict_structure.py` | `benchmark/point_cloude/models/backbone/x3d_utils/explicit_structure.py` |
| `openpoints/models/backbone/X_3D_utils/neighbor_context.py` | `benchmark/point_cloude/models/backbone/x3d_utils/neighbor_context.py` |
| `openpoints/models/classification/cls_base.py` (ClsHead) | `benchmark/point_cloude/models/cls_head.py` |
| `openpoints/dataset/density/density_loader.py` | `benchmark/point_cloude/data/energy_dataset.py` |
| `examples/regression/main.py` | `benchmark/point_cloude/train_energy.py` |
| `cfgs/energy/pointmetabase-s-x-3d.yaml` | `benchmark/point_cloude/cfgs/energy_x3d.yaml` |

---

## 训练配置（对标论文）

| 超参数 | 本项目 | EDBench 论文 |
|--------|--------|-------------|
| 输入点数 | 2048 | 2048 |
| Width | 32 | 32 |
| Stages | 6 | 6 |
| Ball Query K | 32 | 32 |
| Base radius | 0.15 Bohr | 0.15 |
| X-3D stages | {3, 4} | {3, 4} |
| 损失函数 | 标准化目标上的 MSELoss | MSELoss |
| 优化器 | AdamW | AdamW |
| 学习率 | 1e-3 | 1e-3 |
| Weight decay | 0.05 | 0.05 |
| Batch size | 32 | 32 |
| Epochs | 100 | 100 |
| 调度器 | CosineAnnealingLR | CosineAnnealingLR |
| 梯度裁剪 | 1.0 | 1.0 |
| Dropout | 0.5 | 0.5 |
| 数据划分 | scaffold_split | scaffold_split |

---

## 使用方法

### 目录结构

```text
benchmark/point_cloude/
├── data/
│   └── energy_dataset.py       # EDBenchEnergyDataset（FPS 采样 + 能量标签）
├── models/
│   ├── cls_head.py             # RegressionHead（MLP）
│   └── backbone/
│       ├── pointmetabase_x3d.py  # X-3D 主网络
│       └── x3d_utils/
│           ├── explicit_structure.py  # PCA + PointHop
│           └── neighbor_context.py    # 动态权重聚合
├── cfgs/
│   └── energy_x3d.yaml         # 超参数配置
└── train_energy.py             # 训练入口
```

### 冒烟测试（已验证，CPU，~35 分钟）

使用 `-m` 方式从项目根目录运行：

```bash
python -m benchmark.point_cloude.train_energy \
    --pkl_path data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv_path data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache_dir data/ed_energy_5w/cache_fps \
    --max_samples 128 \
    --npoint 512 \
    --epochs 2 \
    --device cpu
```

**实际运行输出**（2 epoch，128 样本，npoint=512，CPU；2026-04-16 复测）：

```text
PointMetaBase-S-X3D  params: 2,478,476
Epoch    1/2  train_loss=1.3050  val_mean_MAE=228.5966  lr=5.00e-04
  E1_Final              MAE=168.3671  RMSE=209.3864  r=-0.0245
  E2_NucRepul           MAE=223.6611  RMSE=269.2984  r=0.1765
  E3_OneElec            MAE=542.8998  RMSE=677.9306  r=0.0668
  E4_TwoElec            MAE=244.3651  RMSE=307.6121  r=0.1202
  E5_XC                 MAE=11.2410   RMSE=13.9821   r=0.0748
  E6_Total              MAE=181.0454  RMSE=217.9585  r=0.1261
Epoch    2/2  train_loss=1.1138  val_mean_MAE=226.8407  lr=0.00e+00
  E1_Final              MAE=165.3687  RMSE=206.0115  r=0.0691
  ...
=== Test Set Evaluation ===
  Mean MAE: 261.4900
Done. Best val mean MAE: 226.8407
```

> **说明**：仅 2 epoch + 128 样本未收敛，MAE 偏高符合预期。由于现在训练时对目标做了 z-score，`train_loss` 会落在接近 1 的量级；评估指标仍然保持原始 Hartree 单位。相对顺序正确（E5 MAE 最小，E3 最大），管道端到端验证通过。

### 完整训练（GPU，推荐）

```bash
python -m benchmark.point_cloude.train_energy \
    --pkl_path data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv_path data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache_dir data/ed_energy_5w/cache_fps \
    --npoint 2048 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 100 \
    --output_dir benchmark/outputs/checkpoints/energy \
    --device cuda \
    --wandb \
    --wandb_project ed-energy \
    --run_name repro-x3d-v1
```

> **注意**：请从项目根目录以 `-m` 方式运行，而非直接 `python benchmark/point_cloude/train_energy.py`，以避免相对包导入错误。
>
> **补充**：训练脚本保存的 checkpoint 中包含 `target_mean` 和 `target_std`，推理或离线评估时需要用它们将模型输出反标准化回 Hartree。

---

## 论文基准结果（对比目标）

论文使用 **scaffold_split**，EDBench 完整 3.3M 数据集，三次运行平均：

| 能量成分 | X-3D MAE (Hartree) ± std |
|---------|--------------------------|
| E1 Final Energy | 190.77 ± 1.98 |
| E2 Nuclear Repulsion | 109.21 ± 2.82 |
| E3 One-Electron | 369.88 ± 1.34 |
| E4 Two-Electron | 150.05 ± 0.27 |
| E5 DFT XC | **8.13 ± 0.51** |
| E6 Total Energy | 190.77 ± 1.98 |

> **注意**：本项目使用 EDBench 的 47k 子集（非完整 3.3M）。由于训练数据量差异，绝对 MAE 数值可能偏高；但相对顺序（E5 MAE 最小，E3 最大）应保持一致。

---

## 评测指标

每种能量分量报告 4 个指标（对标论文 Table 3）：

- **MAE**：平均绝对误差（主要指标）
- **RMSE**：均方根误差
- **Pearson r**：线性相关系数
- **Spearman ρ**：秩相关系数

---

## 结果分析

训练完成后可用各自的分析脚本生成可视化图表和汇总数据。

### 点云模型分析（`benchmark/point_cloude/analyze_results.py`）

```bash
python -m benchmark.point_cloude.analyze_results \
    --run-dir  benchmark/outputs/checkpoints/benchmark/repro-x3d-v1 \
    --pkl-path data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache-dir data/ed_energy_5w/cache_fps \
    --output-dir benchmark/outputs/analysis/pointcloud \
    --device cpu \
    --batch-size 32
```

#### 输出文件

| 文件 | Phase | 内容 |
|------|-------|------|
| `pc_training_curves.png` | A | train_loss（log）+ val MAE + LR vs epoch；需 `metrics.json`（旧 checkpoint 跳过）|
| `pc_scatter.png` | C | 全 split 叠加散点图（pred vs true，2×3） |
| `pc_scatter_{train/valid/test}.png` | C | 各 split 单独散点图 |
| `pc_residual_hist.png` | D1 | test split 残差直方图 + KDE（2×3，每目标一格） |
| `pc_error_vs_density.png` | D2 | test split 绝对误差 vs 分子点云均值密度（Spearman ρ 注释） |
| `pc_error_vs_size.png` | D3 | test split 绝对误差 vs 空间尺寸（bounding box 对角线，Bohr） |
| `pc_error_correlation.png` | D4 | 6 个目标绝对误差的 6×6 Pearson 相关热图 |
| `pc_cdf.png` | E | test split 累积误差分布（CDF），6 条曲线 |
| `analysis_summary.json` | F | per-split per-target MAE / RMSE / Pearson r / Spearman ρ |

> **旧格式 checkpoint 兼容**：`repro-x3d-v1/best.pt` 使用旧 key（`"model"/"args"`），脚本自动检测并兼容；因无 `metrics.json`，Phase A 训练曲线跳过，其余 8 个输出文件正常生成。

#### 各 Phase 分析说明

- **Phase A**：训练过程监控。双 y 轴：左 train_loss（对数刻度），右 val_mean_MAE；灰色虚线标注 best epoch。
- **Phase C**：预测质量核心图。散点越靠近 `y=x` 对角线越好；注释框同时显示 MAE / RMSE / Pearson r / **Spearman ρ**。
- **Phase D1**：偏差（bias）分析。残差均值接近 0 表示模型无系统性偏移；KDE 曲线反映误差分布形态。
- **Phase D2**：误差与密度相关性。Spearman ρ 显著正值说明高密度区域预测更难。
- **Phase D3**：误差与分子大小相关性。空间尺寸大的分子（更重）误差是否更大。
- **Phase D4**：各目标误差相关性热图。E1/E6 误差理论上应 100% 相关（两者物理等价）。
- **Phase E**：CDF 曲线。读取 P(|error| < x) 在各 x 处的值，直观比较不同能量目标的误差尺度。


---

### 结果目录

```
benchmark/outputs/analysis/
├── pointcloud/     ← 点云模型分析结果
└── voxel/          ← 体素模型分析结果
```

---

## 参考文献

```
EDBench: A Large-Scale Electron Density Dataset for Molecular Modeling
Hongxin Xiang et al., NeurIPS 2025
arXiv: 2505.09262
GitHub: https://github.com/HongxinXiang/EDBench
```
