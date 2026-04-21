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
               │     初始化（仅一次）：
               │       原始节点特征嵌入 → shared_state (N, 80)
               │     每轮 B-Block（shared_state 和 p_prev 跨轮传递）：
               │       T_init（shared_state → local chart 状态）
               │       ExplicitStructureEncoder（共享权重，ES → 调制特征）
               │       T_local_msg × num_local_steps（FCLC 内消息传递，默认 2 步）
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

### B-Block 迭代（K 轮，消息传递权重独立，结构编码器共享）

初始化（仅在第 1 轮前执行一次）：原始物理节点特征嵌入 → `shared_state`

每轮按以下顺序执行，`shared_state`（节点级）和 `p_prev`（chart 级）跨轮传递：

```
shared_state（N, 64+8×2）+ p_prev（A, 64+8×2）— 跨轮传递
    │
    ▼ FCLCLocalBlock
    │   T_init：shared_state → local chart 状态
    │   ExplicitStructureEncoder（共享权重）：ES → 调制特征 + chart 状态起点
    │   p_prev 残差叠加（若非首轮）
    │   T_local_msg × num_local_steps（默认 2 步，每步后刷新 ES 编码）
    │   → node_state_shared_next (N, 80)
    │   → local_state_final      (N_M, ...)
    │   → p_next_local           (A, 80)    ← chart 标量(64) + 向量(8×2)
    │
    ▼ IntraLevelBlock（T_intra）
    │   p_bar  (A, 80)  — 层内 chart 图消息传递后
    │
    ▼ InterLevelBlock（T_inter）
        p_new  (A, 80)  — 跨层传播后，作为 p_prev 传入下一轮 B-Block
```

> **Gradient Checkpointing**：`BBlockConfig.use_gradient_checkpointing=True`（默认开启）在训练时用 `torch.utils.checkpoint` 重算各 BBlock 激活，以约 10% 额外计算换取 ~35% 显存节省，可在相同 GPU 上将 per-GPU batch size 翻倍。推理时自动关闭。

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
pip install wandb plotly  # 可选：wandb 训练监控；plotly 可视化
```

---

### 6.2 端到端预处理

**文件**：`scripts/preprocess_edbench.py`

```bash
python scripts/preprocess_edbench.py \
    --pkl      data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv      data/ed_energy_5w/raw/ed_energy_5w.csv \
    --data-dir data/ed_energy_5w \
    --workers  8
```

**必填参数**

| 参数 | 说明 |
|------|------|
| `--pkl` | 原始 PKL 文件路径（mol_EDthresh0.05_data.pkl，~9 GB） |
| `--csv` | 能量标签 CSV 路径（ed_energy_5w.csv） |
| `--data-dir` | 输出根目录（cache_manifold/、packed_stage3/ 等自动创建于此） |

**跳过与续算**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--skip-stage1` | False | 跳过 Stage 1（当 `cache_manifold/all_nl*.pkl` 已存在时自动跳过） |
| `--skip-stage23` | False | 跳过 Stage 2+3（当 `packed_stage3/manifest.json` 已存在时自动跳过） |

Stage 2+3 支持中断续算：若 `packed_stage3/` 下已有 `shard_0000/` 等完整分片，重新运行时会自动跳过已完成的分子，从断点继续写入新分片。

**并行控制**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--workers` | `cpu_count // 4` | 进程数 |
| `--threads-per-proc` | 4 | 每进程内部线程数（Stage 2+3） |
| `--chunksize` | 4 | `imap_unordered` 批量大小 |
| `--shard-size` | 2000 | 每个 packed shard 最大分子数 |

**数据量控制**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-samples` | None | 限制处理分子数（测试用） |

**Stage 1 参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n-levels` | 4 | 等密度层数（百分位阈值数） |
| `--smooth-sigma` | 0.5 | 密度场 Gaussian 预平滑 σ（Bohr） |
| `--min-component-size` | 10 | 最小连通分量顶点数 |

**Stage 2+3 参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tau-r` | 1.0 | chart 半径阈值（Bohr），控制 chart 大小 |
| `--tau-2` | 1.5 | 二阶候选邻居半径倍数 |
| `--local-knn-k` | 12 | chart 内 KNN 图的 K |
| `--chart-knn-k` | 8 | 同层 chart 图的 K |
| `--num-anchors` | 8 | 每个 chart 的 anchor 点数 |
| `--mem-thresh` | None | 预计算全量测地距离矩阵的最大顶点数。高值以 RAM 换速度（消除逐 Dijkstra 调用）；500 GB 内存下建议 15000–20000 |

**Split 参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--split-col` | `scaffold_split` | CSV 中用于划分的列名（`scaffold_split` 或 `random_split`） |

**输出目录结构**

```
data/ed_energy_5w/
├── cache_manifold/all_nl4_s0.50.pkl   ← Stage 1 合并缓存
├── packed_stage3/                      ← Stage 2+3 packed（分片）
│   ├── manifest.json
│   ├── shard_0000/
│   └── ...
├── split.json                          ← train/val/test mol_id 列表
└── energy_stats.json                   ← 训练集能量 z-score 统计量
```

---

### 6.3 单独运行 Stage 2+3（已有 Stage 1 缓存）

**文件**：`scripts/preprocess_stage2_to_packed.py`

```bash
python scripts/preprocess_stage2_to_packed.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --packed-dir   data/ed_energy_5w/packed_stage3 \
    --workers 8 --mem-thresh 15000
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--manifold-pkl` | — | 合并的 Stage 1 PKL（与 `--manifold-cache-dir` 二选一） |
| `--manifold-cache-dir` | — | Stage 1 逐分子缓存目录（与 `--manifold-pkl` 二选一） |
| `--packed-dir` | 必填 | 输出 packed 目录 |
| `--shard-size` | 2000 | 每个 shard 最大分子数 |
| `--workers` | `cpu_count // 4` | 进程数 |
| `--threads-per-proc` | 4 | 每进程内部线程数 |
| `--parallel-mode` | `process` | 并行模式（`process` 或 `thread`） |
| `--n-levels` | 4 | Stage 1 层数（用于定位逐分子缓存文件名） |
| `--smooth-sigma` | 0.5 | Stage 1 平滑 σ（同上） |
| `--tau-r` | 1.0 | chart 半径阈值 |
| `--tau-2` | 1.5 | 二阶邻居半径倍数 |
| `--local-knn-k` | 12 | chart 内 KNN K |
| `--chart-knn-k` | 8 | 同层 chart 图 K |
| `--num-anchors` | 8 | anchor 点数 |
| `--mem-thresh` | None | 测地距离矩阵预计算阈值 |
| `--max-samples` | None | 限制分子数 |
| `--chunksize` | 4 | imap 批量大小 |
| `--resume` | False | 续算：跳过 packed_dir 中已写入的分子 |

---

### 6.4 预处理质量分析

**文件**：`scripts/analyze_preprocessing.py`

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

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--packed-dir` | 必填 | packed_stage3 目录 |
| `--out-dir` | `<packed-dir>/../preprocess_analysis` | 输出目录 |
| `--n-mols` | 500 | 分析的分子数（随机采样） |
| `--plot-mols` | — | 指定生成 3D chart-graph HTML 的 mol_id 列表 |
| `--quick-check` | False | 仅运行 Pass/Fail 检查，跳过所有图表 |
| `--seed` | 42 | 随机采样种子 |

输出：`summary_stats.json`、`per_mol_stats.csv`、`flagged_mols.json`，以及 11 项分布直方图和交互式 3D chart 图。

---

### 6.5 前向冒烟测试

**文件**：`scripts/smoke_stage6_forward.py`

```bash
python scripts/smoke_stage6_forward.py \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --n-mols 4 --device cpu --num-bblocks 1
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--packed-dir` | 必填 | packed_stage3 目录 |
| `--n-mols` | 4 | 测试分子数 |
| `--device` | `cpu` | 运行设备 |
| `--num-bblocks` | 1 | B-Block 数（快速测试用 1） |

验证项：`energy_pred.shape == (B, 6)`，无 NaN/Inf，每分子每目标每 head 注意力之和为 1。

---

### 6.6 训练 ED2E 模型

**文件**：`scripts/train.py`

支持单卡和多卡 DDP 两种方式，自动检测 `LOCAL_RANK` 环境变量（由 `torchrun` 注入）。

#### 单卡训练

```bash
python scripts/train.py \
    --data-dir  data/ed_energy_5w \
    --csv-path  data/ed_energy_5w/raw/ed_energy_5w.csv \
    --out-dir   runs/ed2e_k3 \
    --num-bblocks 3 --batch-size 8 --epochs 100 --device cuda \
    --num-workers 4 \
    --wandb --wandb-project ed2e --wandb-run-name k3_v1
```

#### 多卡 DDP（推荐，8× H200）

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --data-dir  data/ed_energy_5w \
    --csv-path  data/ed_energy_5w/raw/ed_energy_5w.csv \
    --out-dir   runs/ed2e_ddp \
    --num-bblocks 3 --batch-size 8 --grad-accum 2 \
    --epochs 100 --num-workers 4 \
    --wandb --wandb-project ed2e --wandb-run-name ddp_k3_v1
# 等效 batch = 8 × 8 × 2 = 128
# bf16 AMP 和 gradient checkpointing 默认开启
```

#### 断点续训

```bash
# 单卡
python scripts/train.py ... --resume runs/ed2e_k3/last.pt

# 多卡
torchrun --nproc_per_node=8 scripts/train.py ... --resume runs/ed2e_ddp/last.pt
```

**必填参数**

| 参数 | 说明 |
|------|------|
| `--data-dir` | 数据根目录（含 packed_stage3/、split.json、energy_stats.json） |
| `--csv-path` | 能量标签 CSV 路径 |
| `--out-dir` | Checkpoint 和日志输出目录 |

**模型结构**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-bblocks` | 3 | B-Block 迭代轮数 K |
| `--num-heads` | 4 | 读出层多头注意力头数 |
| `--head-hidden` | 128 | EnergyHead MLP 隐层维度 |
| `--dropout` | 0.1 | EnergyHead Dropout 率 |

**训练超参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 8 | 每 GPU 批次大小 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-3 | AdamW 学习率 |
| `--weight-decay` | 1e-4 | AdamW 权重衰减 |
| `--grad-accum` | 1 | 梯度累积步数（等效 batch = batch_size × GPUs × N） |
| `--num-workers` | 4 | 每进程 DataLoader 工作进程数 |
| `--device` | `cuda` | 单卡模式设备（DDP 时由 torchrun 管理，无需指定） |
| `--resume` | — | 续训 checkpoint 路径（`.pt` 文件） |
| `--log-every` | 50 | 每 N 个 optimizer step 打印一次 step-level 损失 |

**精度与显存**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--no-amp` | False | 禁用 bf16 AMP（默认开启，仅 CUDA 有效） |

> Gradient Checkpointing 由 `BBlockConfig.use_gradient_checkpointing`（默认 `True`）控制，训练时自动启用，无需命令行参数。

**W&B 监控**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--wandb` | False | 启用 Weights & Biases（需 `pip install wandb`） |
| `--wandb-offline` | False | 离线模式：日志写入本地磁盘，不需要网络；训练后用 `wandb sync` 上传 |
| `--wandb-project` | `ed2e` | W&B 项目名 |
| `--wandb-entity` | — | W&B 团队/用户名 |
| `--wandb-run-name` | out-dir 末段 | Run 名称 |
| `--wandb-tags` | — | 空格分隔的标签列表 |

离线模式使用方法：
```bash
# 训练时记录到本地
torchrun --nproc_per_node=8 scripts/train.py ... \
    --wandb --wandb-offline --wandb-project ed2e

# 训练完成后（有网络时）同步到 W&B 服务器
wandb sync runs/ed2e_ddp/wandb/offline-run-*
```

**DDP 注意事项**

- 由 `torchrun` 启动时自动检测 `LOCAL_RANK`，初始化 NCCL 进程组
- Checkpoint、日志和 W&B 只在 **rank 0** 进程执行
- Validation 只在 **rank 0** 运行（避免重复统计）
- 需要 `/dev/shm` ≥ 4–8 GB（Docker 容器默认 64 MB 需手动扩容：`--shm-size=8g`）

**Checkpoint 格式**

```python
{
    "config":     ED2EConfig,          # dataclass，含完整超参数
    "state_dict": model.state_dict(),  # 原始模型权重（非 DDP 包装）
    "norm_mean":  np.ndarray (6,),     # 训练集能量均值
    "norm_std":   np.ndarray (6,),
    "epoch":      int,
    "val_mae":    np.ndarray (6,),     # per-target MAE（Hartree）
}
```

加载：
```python
ckpt  = torch.load("best.pt", map_location="cpu")
model = ED2EModel(ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
norm_stats = {"mean": ckpt["norm_mean"], "std": ckpt["norm_std"]}
```

---

### 6.7 注意力可视化

**文件**：`scripts/visualize_attention.py`

```bash
python scripts/visualize_attention.py \
    --checkpoint runs/ed2e_k3/best.pt \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --mol-ids 308 42 100 \
    --out-dir  figs/attention
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 必填 | 训练好的 `.pt` checkpoint 路径 |
| `--packed-dir` | 必填 | packed_stage3 目录 |
| `--out-dir` | 必填 | 输出目录 |
| `--mol-ids` | — | 指定 mol_id 列表（与 `--n-mols` 互斥） |
| `--n-mols` | — | 取前 N 个分子（与 `--mol-ids` 互斥） |
| `--device` | `cpu` | 运行设备 |

输出：每分子每目标的交互式 3D chart 散点图（颜色 = 注意力权重）、`attention_data.csv`、`attn_by_level.png`。

---

## 7. FCLC 超参数消融

**文件**：`scripts/ablate_fclc_chart_size.py`

```bash
python scripts/ablate_fclc_chart_size.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --n-mols 50 \
    --output-csv data/fclc_ablation.csv
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--manifold-pkl` | 必填 | 合并的 Stage 1 manifold PKL |
| `--n-mols` | 50 | 使用的分子数（取前 N 个，确定性） |
| `--tau-r` | `0.5 0.75 1.0 1.25 1.5` | 扫描的 tau_r 值列表（空格分隔） |
| `--tau-2` | `1.0 1.5 2.0` | 扫描的 tau_2 值列表（空格分隔） |
| `--min-chart-size` | 5 | 最小 chart 大小过滤阈值 |
| `--output-csv` | `fclc_ablation.csv` | 输出 CSV 路径 |

输出 CSV 每行一种 `(tau_r, tau_2)` 组合，记录 chart 数量均值/标准差、chart 大小分布、覆盖率等。

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
