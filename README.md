# ED2E — Electron Density to Energy

## 1. 项目概述

根据 Hohenberg-Kohn 定理，分子的全部基态性质由其电子密度唯一决定。**ED2E** 的目标是验证这一理论的可学习性：**仅凭三维电子密度场，能否精确预测 DFT 计算得到的 6 种能量分量？**

本项目的核心创新是在电子密度的等密度面（isodensity manifold）上构建分层几何表示，通过 **FCLC（Feature-Compatible Local Chart）图谱** 将连续场转化为 GNN 可消费的拓扑结构，并用 **BBlock 迭代神经网络**（局部聚合 + 层内消息传递 + 层间消息传递 × K 轮）完成特征提取，最终用多头交叉注意力读出 6 维能量预测。

### 预测目标（6 种能量，单位 Hartree）

| 索引 | 名称 | 含义 |
|------|------|------|
| E1 | DF-RKS Final Energy | DFT 最终总能量 |
| E2 | Nuclear Repulsion | 核间排斥能 |
| E3 | One-Electron Energy | 动能 + 电子-核吸引能 |
| E4 | Two-Electron Energy | 电子-电子排斥能 |
| E5 | DFT XC Energy | 交换相关能 |
| E6 | Total Energy | 各分量之和（= E1） |

### 数据集

- **分子数**：47,986（EDBench 47k 子集，scaffold split）
- **密度文件**：`data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl`（~9 GB，密度阈值 0.05 a.u.）
- **标签文件**：`data/ed_energy_5w/raw/ed_energy_5w.csv`（SMILES + 6 种能量 + split 列）

---

## 2. 目录结构

```
ED2E/
├── ed2e/                     ← Python 主包
│   ├── data/                 — 数据结构、I/O、各阶段构建函数
│   │   ├── manifold.py       — Stage 1：等密度流形提取
│   │   ├── fclc.py           — Stage 2：FCLC 图谱构建
│   │   ├── stage3_local.py   — Stage3Sample dataclass + 构建函数
│   │   ├── stage3_packed.py  — packed mmap 数据集
│   │   ├── energy_stats.py   — 能量 z-score 统计量
│   │   ├── dataset.py        — 遗留 EDBenchPKLDataset（点云/体素用）
│   │   └── clustering.py     — 密度加权 K-Means
│   ├── model/                — 神经网络各阶段
│   │   ├── stage3_local.py   — FCLCLocalBlock（T_local_msg × 3 + T_chart_encode）
│   │   ├── stage4_intra.py   — IntraLevelBlock（T_intra）
│   │   ├── stage5_inter.py   — InterLevelBlock（T_inter）
│   │   ├── bblock.py         — BBlock + ED2EBBlockStack
│   │   ├── readout.py        — MultiHeadChartReadout + EnergyHeads
│   │   └── ed2e.py           — ED2EModel（完整端到端模型）
│   └── utils/                — 流形与 FCLC 可视化工具
├── scripts/                  ← 预处理、训练、分析脚本
│   ├── preprocess_edbench.py — 端到端预处理（Stage 1→5 + split + stats）
│   ├── preprocess_stage1.py  — 单独运行 Stage 1
│   ├── preprocess_stage2.py  — 单独运行 Stage 2
│   ├── preprocess_stage3.py  — 单独运行 Stage 3 + 4 + 5
│   ├── pack_stage3_cache.py  — 打包为 packed mmap 格式
│   ├── preprocess_stage2_to_packed.py  — Stage 2→packed 一体化流水线
│   ├── train.py              — ED2E 模型训练
│   ├── analyze_preprocessing.py  — 预处理质量分析（11 项指标）
│   ├── visualize_attention.py    — 注意力权重可视化
│   ├── ablate_fclc_chart_size.py — FCLC 超参数消融
│   ├── smoke_stage3_local_forward.py  — Stage 3 前向验证
│   ├── smoke_stage4_intra_forward.py  — Stage 4 前向验证
│   ├── smoke_stage5_inter_forward.py  — Stage 5 前向验证
│   └── smoke_stage6_forward.py        — 完整模型前向验证
├── benchmark/                ← 对照基线模型
│   ├── point_cloude/         — PointMetaBase-S-X3D 点云基线
│   └── voxel/                — 3D DenseNet 体素基线
├── data/                     ← 数据目录（不纳入版本控制）
│   └── ed_energy_5w/
│       ├── raw/              — ed_energy_5w.csv
│       ├── processed/        — mol_EDthresh0.05_data.pkl
│       ├── cache_manifold/   — Stage 1 缓存
│       ├── packed_stage3/    — Stage 3 packed mmap 缓存
│       ├── split.json        — train/val/test mol_id 列表
│       └── energy_stats.json — 训练集能量均值/标准差
├── docs/                     ← 各阶段详细设计文档
└── tests/                    ← 单元/集成测试
```

---

## 3. 完整数据流

```
mol_EDthresh0.05_data.pkl  +  ed_energy_5w.csv
         │
         ▼ scripts/preprocess_edbench.py（或分阶段脚本）
         │
         ├── [Stage 1] extract_manifold_levels()
         │     └── ManifoldLevel × 4（等密度层）
         │           └── ManifoldComponent（连通分量，三角网格 + 每顶点特征）
         │                 顶点：coords(V,3) + normals(V,3)
         │                       scalar(V,5): ‖∇ρ‖, Δρ, H, K, ∂²_nρ
         │                       vector(V,2,3): ∇_M‖∇ρ‖, ∇_M H
         │
         ├── [Stage 2] build_fclc_levels()
         │     └── FCLCLevel × 4
         │           └── FCLCChart（重叠局部 chart P_a）
         │                 center(3), frame(2,3), local_coords(V_a,2)
         │                 quadrant(V_a), es_local(53)
         │                 inter_weights（跨层方向权重）
         │
         ├── [Stage 3] build_stage3_sample()
         │     └── Stage3Sample（扁平化张量，~35 个字段）
         │           节点：node_xyz(N,3), node_scalar_raw(N,5), …
         │           归属：chart_membership(M,2), local_coords(M,2), …
         │           边：local_knn_edge_index(2,E_loc)
         │               chart_graph_edge_index(2,E_cg)  ← Stage 4
         │               overlap_edge_index(2,E_ov)       ← Stage 4
         │               inter_level_edge_index(2,E_inter) ← Stage 5
         │
         ├── [Pack] pack_stage3_cache() / Stage3ShardedWriter
         │     └── packed_stage3/
         │           ├── manifest.json（shard 列表）
         │           ├── shard_XXXX/
         │           │     ├── meta.json
         │           │     ├── index.npz（mol_ids + 各实体指针数组）
         │           │     └── {field}.npy × ~35（mmap，训练时切片读取）
         │           ├── split.json
         │           └── energy_stats.json
         │
         ▼ scripts/train.py
         │
         ├── Stage3PackedDataset.__getitem__()
         │     → Stage3Sample → collate → Stage3TensorBatch
         │
         └── ED2EModel.forward(batch)
               ├── ED2EBBlockStack × K（默认 K=3）
               │     每轮 B-Block：
               │       T_init（node → chart 初始化）
               │       T_local_msg × 3（FCLC 内消息传递）
               │       T_chart_encode（chart state 编码）
               │       T_intra（层内 chart 图，IntraLevelBlock）
               │       T_inter（层间 chart 图，InterLevelBlock）
               ├── MultiHeadChartReadout（6目标 × 4 head 交叉注意力）
               └── EnergyHeads × 6（独立 MLP）
                     → (B, 6)  z-score 归一化空间
                     → 反归一化 → Hartree
```

---

## 4. 网络架构

### 输入

`Stage3TensorBatch`，主要字段：

| 字段 | 形状 | 说明 |
|------|------|------|
| `node_xyz` | (N_total, 3) | 所有分子所有层顶点坐标（Bohr） |
| `node_scalar_raw` | (N_total, 5) | 5 个标量物理特征 |
| `node_vector_raw` | (N_total, 2, 3) | 2 个向量物理特征 |
| `chart_membership` | (M_total, 2) | [chart_id, node_id] 归属关系 |
| `local_knn_edge_index` | (2, E_loc) | chart 内 KNN 图边 |
| `chart_graph_edge_index` | (2, E_cg) | 同层 chart 间边 |
| `overlap_edge_index` | (2, E_ov) | 重叠 chart 对边 |
| `inter_level_edge_index` | (2, E_inter) | 跨层 chart 边 |
| `inter_level_weights` | (E_inter,) | 跨层方向权重（counting-based） |
| `node_batch` / `chart_batch` | (N/A_total,) | 批次索引 |

### B-Block 迭代（K 轮，权重独立）

每轮按以下顺序执行：

```
shared_state（N, 64+8×2）— 跨轮传递
    │
    ▼ FCLCLocalBlock（T_init + T_local_msg × 3 + T_chart_encode）
    │   node_state_shared_next (N, 80)
    │   local_state_final      (N_M, ...)
    │   p_next_local           (A, 80)    ← chart 标量(64) + 向量(8×2)
    │
    ▼ IntraLevelBlock（T_intra）
    │   p_bar  (A, 80)  — 层内 chart 图消息传递后
    │
    ▼ InterLevelBlock（T_inter）
        p_new  (A, 80)  — 跨层传播后，传入下一轮 B-Block
```

### 读出与能量头

```
p_new (A, 80)
    │
    ▼ MultiHeadChartReadout（cross-attention）
    │   query: target_queries (6, 4, 24)  — 可学习，每目标独立
    │   K, V: 共享跨目标，来自 chart 特征投影
    │   segment_softmax：每分子内 chart 权重归一化
    │   global_features (B, 6, 96)
    │
    ▼ EnergyHeads × 6（完全独立权重 MLP）
        Linear(96→128) → LayerNorm → GELU → Dropout(0.1) → Linear(128→1)
        → energy_pred (B, 6)  z-score 归一化空间
```

### 参数规模（默认配置 K=3）

| 组件 | 参数量（约） |
|------|------------|
| FCLCLocalBlock × 3 | ~1.3M |
| IntraLevelBlock × 3 | ~0.7M |
| InterLevelBlock × 3 | ~0.5M |
| MultiHeadChartReadout | ~0.1M |
| EnergyHeads × 6 | ~0.07M |
| **合计** | **~2.7M** |

---

## 5. 预处理数据结构速查

| 阶段 | 主要 dataclass | 关键字段 | 详细文档 |
|------|--------------|---------|--------|
| Stage 1 | `ManifoldLevel` → `ManifoldComponent` | `verts(V,3)`, `scalar_features(V,5)`, `vector_features(V,2,3)` | [docs/stage1_manifold_extraction.md](docs/stage1_manifold_extraction.md) |
| Stage 2 | `FCLCLevel` → `FCLCChart` | `center(3)`, `frame(2,3)`, `es_local(53)`, `inter_weights` | [docs/stage2_fclc_construction.md](docs/stage2_fclc_construction.md) |
| Stage 3 | `Stage3Sample` | ~35 个 numpy 字段，见详细文档 | [docs/stage3_fclc_local_aggregation.md](docs/stage3_fclc_local_aggregation.md) |
| Packed | `Stage3PackedDataset` | mmap `.npy` × ~35 + `index.npz` | [docs/stage3_fclc_local_aggregation.md](docs/stage3_fclc_local_aggregation.md) |
| 附属 | `split.json` / `energy_stats.json` | train/val/test mol_id 列表；均值、标准差 | [docs/stage6_output_head.md](docs/stage6_output_head.md) |

---

## 6. 快速上手

### 6.1 安装依赖

```bash
pip install torch numpy scikit-learn "scikit-image>=0.19" "scipy>=1.7" numba tqdm
pip install plotly  # 可选，仅可视化需要
```

### 6.2 端到端预处理（一条命令）

```bash
python scripts/preprocess_edbench.py \
    --pkl  data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv  data/ed_energy_5w/raw/ed_energy_5w.csv \
    --data-dir data/ed_energy_5w \
    --workers 8
```

输出目录结构：

```
data/ed_energy_5w/
├── cache_manifold/all_nl4_s0.50.pkl   ← Stage 1
├── packed_stage3/                      ← Stage 3 packed（分片）
│   ├── manifest.json
│   ├── shard_0000/
│   └── ...
├── split.json                          ← scaffold_split 划分
└── energy_stats.json                   ← 训练集能量 z-score 统计量
```

可选参数：

```bash
# 快速冒烟测试（前 200 个分子）
python scripts/preprocess_edbench.py ... --max-samples 200

# 跳过已完成的 Stage 1（恢复运行）
python scripts/preprocess_edbench.py ... --skip-stage1

# 调整超参数
python scripts/preprocess_edbench.py ... \
    --n-levels 4 --smooth-sigma 0.5 \
    --tau-r 1.0 --tau-2 1.5 \
    --local-knn-k 12 --chart-knn-k 8 --num-anchors 8
```

### 6.3 预处理质量分析

```bash
python scripts/analyze_preprocessing.py \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --out-dir    data/ed_energy_5w/preprocess_analysis \
    --n-mols 1000 --plot-mols 308 42 100

# 快速 Pass/Fail 检查（不画图）
python scripts/analyze_preprocessing.py \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --quick-check --n-mols 200
```

分析输出：`summary_stats.json`、`per_mol_stats.csv`、`flagged_mols.json`，以及 11 项分布直方图和交互式 3D chart 图。

### 6.4 前向冒烟测试

```bash
# 完整模型端到端验证（CPU，4 个分子，1 个 B-Block）
python scripts/smoke_stage6_forward.py \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --n-mols 4 --device cpu --num-bblocks 1
```

验证项：`energy_pred.shape == (4, 6)`，无 NaN/Inf，注意力权重之和为 1。

### 6.5 训练 ED2E 模型

```bash
python scripts/train.py \
    --data-dir  data/ed_energy_5w \
    --csv-path  data/ed_energy_5w/raw/ed_energy_5w.csv \
    --out-dir   runs/ed2e_k3 \
    --num-bblocks 3 \
    --batch-size 8 \
    --epochs 100 \
    --device cuda
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-bblocks` | 3 | B-Block 迭代轮数 |
| `--batch-size` | 8 | 批次大小 |
| `--lr` | 1e-3 | 学习率 |
| `--weight-decay` | 1e-4 | AdamW 权重衰减 |
| `--epochs` | 100 | 训练轮数 |
| `--device` | cpu | 训练设备 |
| `--resume` | — | 断点续训路径 |

Checkpoint 格式：

```python
{
    "config":     ED2EConfig,       # dataclass
    "state_dict": model.state_dict(),
    "norm_mean":  np.ndarray (6,),  # 训练集能量均值
    "norm_std":   np.ndarray (6,),
    "epoch":      int,
    "val_mae":    np.ndarray (6,),  # per-target MAE（Hartree）
}
```

### 6.6 注意力可视化

```bash
python scripts/visualize_attention.py \
    --checkpoint runs/ed2e_k3/best.pt \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --mol-ids 308 42 100 \
    --out-dir  figs/attention
```

输出每个分子每个能量目标的交互式 3D chart 散点图（颜色 = 注意力权重），以及 `attention_data.csv` 和 `attn_by_level.png`。

---

## 7. FCLC 超参数消融

通过扫描 `(tau_r, tau_2)` 网格来确定 chart 大小的最优配置：

```bash
python scripts/ablate_fclc_chart_size.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --n-mols 50 \
    --output-csv data/fclc_ablation.csv
```

输出 `fclc_ablation.csv`：每行一种 `(tau_r, tau_2)` 组合，记录 chart 数量均值/标准差、chart 大小分布、覆盖率等统计量。

---

## 8. 基线 Benchmark

### 8.1 点云 PointMetaBase-S-X3D

基于 EDBench 原始 X-3D 模型，输入为 FPS 采样后的 2048 点点云 `(B, 2048, 4)`（xyz + 密度）。

```bash
# 训练
python -m benchmark.point_cloude.train_energy \
    --pkl-path  data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv-path  data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache-dir data/ed_energy_5w/cache_fps \
    --npoint 2048 --batch-size 32 --epochs 100 --device cuda

# 分析（训练曲线、散点图、残差、CDF 等）
python -m benchmark.point_cloude.analyze_results \
    --run-dir   benchmark/outputs/checkpoints/benchmark/repro-x3d-v1 \
    --pkl-path  data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv-path  data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache-dir data/ed_energy_5w/cache_fps \
    --output-dir benchmark/outputs/analysis/pointcloud \
    --device cpu
```

详见 [docs/benchmark/ED2energy_bench_mark.md](docs/benchmark/ED2energy_bench_mark.md)。

### 8.2 体素 3D DenseNet

基于 ELFNet 结构，将电子密度点云体素化为 `(B, 1, 14, 14, 14)` 网格后输入 3D DenseNet。

```bash
# 预处理（离线体素化）
python -m benchmark.voxel.preprocess_voxel_cache \
    --pkl-path data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache-dir data/ed_energy_5w/cache_voxel

# 训练
python -m benchmark.voxel.train_energy \
    --pkl-path  data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv-path  data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache-dir data/ed_energy_5w/cache_voxel \
    --run-name voxel_full --batch-size 32 --epochs 100 --device cuda

# 分析
python -m benchmark.voxel.analyze_results \
    --run-dir   benchmark/outputs/checkpoints/train/voxel_full_density \
    --csv-path  data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache-dir data/ed_energy_5w/cache_voxel \
    --output-dir benchmark/outputs/analysis/voxel \
    --device cpu
```

详见 [docs/benchmark/voxel_edbench_training.md](docs/benchmark/voxel_edbench_training.md)。

---

## 9. 文档导航

| 文档 | 内容 |
|------|------|
| [docs/pkl_data_structure.md](docs/pkl_data_structure.md) | 原始 PKL 文件格式与字段说明 |
| [docs/stage1_manifold_extraction.md](docs/stage1_manifold_extraction.md) | 等密度流形提取（Marching Cubes + B-spline 导数 + 曲率） |
| [docs/stage2_fclc_construction.md](docs/stage2_fclc_construction.md) | FCLC 图谱构建（测地中心 + 前沿生长 + ES_local 描述子） |
| [docs/stage3_fclc_local_aggregation.md](docs/stage3_fclc_local_aggregation.md) | Stage3Sample schema + packed mmap 格式 + local block |
| [docs/stage4_intra_chart_propagation.md](docs/stage4_intra_chart_propagation.md) | IntraLevelBlock（层内 chart 图消息传递） |
| [docs/stage5_inter_chart_propagation.md](docs/stage5_inter_chart_propagation.md) | InterLevelBlock（跨层 chart 图消息传递） |
| [docs/stage6_output_head.md](docs/stage6_output_head.md) | BBlock 整合 + 多头读出 + 能量头 + 完整训练流程 |
| [docs/benchmark/ED2energy_bench_mark.md](docs/benchmark/ED2energy_bench_mark.md) | 点云 PointMetaBase-S-X3D baseline |
| [docs/benchmark/voxel_edbench_training.md](docs/benchmark/voxel_edbench_training.md) | 体素 3D DenseNet baseline |

---

## 10. 参考文献

```
EDBench: A Large-Scale Electron Density Dataset for Molecular Modeling
Hongxin Xiang et al., NeurIPS 2025
arXiv: 2505.09262
GitHub: https://github.com/HongxinXiang/EDBench
```
