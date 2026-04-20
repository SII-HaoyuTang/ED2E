# Stage 6: B-Block 整合、多头读出与能量预测

## 当前范围

本阶段完成整个 ED2E 模型的最终组装，包括：

- `ED2EBBlockStack`：K 个独立权重 B-block 的循环迭代，`shared_state` 跨轮传递
- `MultiHeadChartReadout`：目标感知多头交叉注意力，对全部 FCLC chart 做池化
- `EnergyHeads`：6 个完全独立权重的输出 MLP
- `ED2EModel` + `ED2EConfig`：端到端封装
- 预处理脚本 `preprocess_edbench.py`：原始 PKL+CSV → 训练就绪缓存
- 预处理质量分析 `analyze_preprocessing.py`
- 训练脚本 `train.py`（EDBench split + z-score 归一化 + Huber loss）
- 注意力可视化 `visualize_attention.py`（BERT 风格）

---

## 在 B-Block 中的位置

```
[T_inter ∘ T_intra ∘ (T_chart_encode ∘ T_local_msg) × num_local_steps ∘ T_chart_encode ∘ T_init] × K
                                                                    ↓
                                              MultiHeadChartReadout (cross-attn)
                                                                    ↓
                                                    EnergyHeads × 6 (独立 MLP)
                                                                    ↓
                                                          energy_pred (B, 6)
```

---

## 一、BBlock 整合

### 文件：`ed2e/model/bblock.py`

#### 数据流

```
ED2EBBlockStack.forward(batch):
  shared_state = _initialize_shared_state(batch)   # 原始特征嵌入，仅调用一次
  p_prev = None
  for block in blocks:
    out3 = FCLCLocalBlock(batch, shared_state=shared_state, p_prev=p_prev)
        → node_state_shared_next (N, 64+8×2)   ← 传给下一轮 T_init
        → local_state_final      (N_M, ...)     ← 节点状态，供 overlap ctx
        → p_next_local           (A, 64+8×2)   ← chart 状态
    p_bar = IntraLevelBlock(p_next_local, local_state_final, batch, intra_static)
        → p_bar  (A, 64+8×2)
    p_new = InterLevelBlock(p_bar, batch, inter_static)
        → p_new  (A, 64+8×2)
    shared_state = node_state_shared_next   # 节点状态跨轮传递
    p_prev = p_new                          # chart 状态跨轮传递
```

`intra_static` 和 `inter_static` 在循环前各预计算一次，在所有 K 轮复用（静态几何）。

`ExplicitStructureEncoder`（ES → 调制特征）在 `ED2EBBlockStack` 中只实例化一份，所有 K 个 BBlock 共享同一套权重。

#### `shared_state` 和 `p_prev` 传递规则

| 轮次 | 输入 `shared_state` | 输入 `p_prev` | 输出 `node_state_shared_next` |
|------|--------------------|--------------|-----------------------------|
| k=0 | `ED2EBBlockStack._initialize_shared_state()` 的输出 | `None` | (N, 64) + (N, 8, 2) |
| k=1 | 上轮 `node_state_shared_next` | 上轮 `p_new` | 同上 |
| k=K-1 | 同上 | 同上 | 丢弃（只取 `p_new`） |

> **注**：`shared_state` 是 per-**NODE**（形状 (N, ...)），初始化由 `ED2EBBlockStack._initialize_shared_state()` 完成（含 `shared_scalar_in` 和 `shared_vector_in` 两个 MLP），不在 `FCLCLocalBlock` 内部初始化。`p_prev` 非 None 时，其值以残差方式叠加到首次 `encode_structure` 的输出上，作为 chart 状态的起点。

#### Config

```python
@dataclass
class BBlockConfig:
    num_bblocks: int = 3
    local_cfg:   Stage3LocalConfig = field(default_factory=Stage3LocalConfig)
    intra_cfg:   Stage4IntraConfig = field(default_factory=Stage4IntraConfig)
    inter_cfg:   Stage5InterConfig = field(default_factory=Stage5InterConfig)
    use_gradient_checkpointing: bool = True   # 训练时重算 BBlock 激活以节省 ~35% 显存
```

`use_gradient_checkpointing=True` 时，`ED2EBBlockStack.forward()` 在训练模式下用 `torch.utils.checkpoint`（`use_reentrant=False`）包装每个 BBlock，以约 10% 额外计算换取 ~35% 显存节省，可在相同显存下将 per-GPU batch size 翻倍。推理时自动关闭（`self.training == False`）。

---

## 二、多头 Chart 交叉注意力读出

### 文件：`ed2e/model/readout.py`，类 `MultiHeadChartReadout`

#### 设计思路

每个能量目标有独立的 query 向量，K 和 V 在所有目标间共享。注意力在每个分子内部通过 `_segment_softmax` 归一化，保证各 chart 权重之和为 1。

#### Config

```python
@dataclass
class ReadoutConfig:
    scalar_dim:    int = 64
    vector_dim:    int = 8
    level_emb_dim: int = 8    # 层级 embedding 维度
    num_levels:    int = 4
    token_dim:     int = 96   # 必须整除 num_heads；head_dim = 96/4 = 24
    num_heads:     int = 4
```

#### 特征构造

```
chart 输入特征（A, 80）：
  [p_new.scalar(64) | p_new.vector.norm(dim=-1)(8) | level_emb(8)]
    ↓ chart_enc: _MLP(80 → 96 → 96)
  H_a  (A, 96)

K = W_k(H_a).view(A, H, d)   [共享跨目标]
V = W_v(H_a).view(A, H, d)
```

> 向量流特征取 `‖·‖`（L2 norm over last dim）而非展平，保持对等变换的不变性。

#### 注意力计算

```
query: target_queries  (T, H, d)  — nn.Parameter，初始 N(0, 1/√d)

score_{a,h,t} = q_t^h · K_{a,h} / √d          (A,)
alpha_{a,h,t} = segment_softmax(score, chart_batch, B)   ∈ [0,1], Σ_a = 1 per mol

g_{t,h} = scatter_add(alpha * V_h, chart_batch, B)      (B, d)
g_t     = concat_h(g_{t,h})                             (B, token_dim=96)
```

#### 输出格式

| 键 | 形状 | 条件 |
|----|------|------|
| `global_features` | `(B, T, 96)` | 始终 |
| `attn_weights` | `(T, H, A)` | `return_attn=True` |

**per-molecule 切分**（可视化时使用）：
```python
attn:       Tensor    # (T, H, A)
chart_batch: LongTensor  # (A,)

mask         = (chart_batch == mol_idx)     # (A,)
attn_mol     = attn[:, :, mask]            # (T, H, A_mol)
attn_per_target = attn_mol.mean(dim=1)    # (T, A_mol)  对 head 平均
```

---

## 三、独立输出头

### 文件：`ed2e/model/readout.py`，类 `EnergyHeads`

```
g: (B, T, token_dim)
  → heads[t](g[:, t, :]) for t in range(T)   各目标完全独立
  → cat → (B, T)                               z-score 归一化空间
```

每个 MLP 结构：`Linear(96→128) → LayerNorm(128) → GELU → Dropout(0.1) → Linear(128→1)`

6 个 MLP 权重完全独立，无参数共享。

---

## 四、完整模型

### 文件：`ed2e/model/ed2e.py`，类 `ED2EModel`

```python
cfg = ED2EConfig()          # num_targets=6 是唯一来源
model = ED2EModel(cfg)

out = model(batch)
# out["energy_pred"]: (B, 6)  z-score 归一化空间

out = model(batch, return_attn=True)
# out["energy_pred"]:    (B, 6)
# out["attn_weights"]:   (T, H, A)
# out["chart_batch"]:    (A,)
# out["chart_level_id"]: (A,)
# out["chart_center"]:   (A, 3)
```

#### `ED2EConfig`

```python
@dataclass
class ED2EConfig:
    bblock:             BBlockConfig  = field(default_factory=BBlockConfig)
    readout:            ReadoutConfig = field(default_factory=ReadoutConfig)
    num_targets:        int   = 6          # 唯一 source of truth
    energy_head_hidden: int   = 128
    energy_dropout:     float = 0.1
    target_names: Tuple[str,...] = TARGET_NAMES   # 6 个能量目标名称
```

`num_targets` 只在 `ED2EConfig` 中定义，`ReadoutConfig` 和 `EnergyHeads` 通过参数接收，不重复定义。

#### 能量目标顺序

| 索引 | 名称 |
|------|------|
| 0 | DF-RKS_Final |
| 1 | Nuclear_Repulsion |
| 2 | One_Electron |
| 3 | Two_Electron |
| 4 | DFT_XC |
| 5 | Total |

与 CSV `label` 列中的空格分隔顺序一致。

---

## 五、能量归一化

### 文件：`ed2e/data/energy_stats.py`

z-score 归一化，统计量仅从训练集计算：

```python
labels = load_energy_labels(csv_path)        # {mol_id: (6,) float32}
splits = load_split_ids(csv_path)            # {"train": [...], "val": [...], "test": [...]}
stats  = compute_energy_stats(labels, train_mol_ids=splits["train"])
# stats["mean"]: (6,) float32
# stats["std"]:  (6,) float32，clamp ≥ 1e-6

save_energy_stats(stats, "data/.../energy_stats.json")
```

> CSV 的 `label` 列是单个空格分隔字符串（6 个值），`index` 列是 mol_id。使用 `csv.DictReader`（stdlib），无 pandas 依赖。

训练时归一化：`(y − mean) / std`
推理/评估时反归一化：`y_norm × std + mean`

---

## 六、端到端预处理

### 文件：`scripts/preprocess_edbench.py`

一条命令从原始数据生成训练就绪缓存：

```
mol_EDthresh0.05_data.pkl  +  ed_energy_5w.csv
  [Stage 1]  → cache_manifold/all_nl4_s0.50.pkl
  [Stage 2+3]→ packed_stage3/  (分片 Stage3PackedDataset)
  [Split]    → split.json       (scaffold_split 或 random_split)
  [Stats]    → energy_stats.json (训练集 z-score 统计量)
```

每步若输出已存在则自动跳过；`--skip-stage1` / `--skip-stage23` 可强制跳过。

```bash
python scripts/preprocess_edbench.py \
    --pkl      data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv      data/ed_energy_5w/raw/ed_energy_5w.csv \
    --data-dir data/ed_energy_5w \
    --workers  8

# 快速测试（前 200 个分子）
python scripts/preprocess_edbench.py ... --max-samples 200

# 恢复：跳过已完成的 Stage 1
python scripts/preprocess_edbench.py ... --skip-stage1
```

**输出目录结构**

```
data/ed_energy_5w/
├── cache_manifold/
│   └── all_nl4_s0.50.pkl
├── packed_stage3/
│   ├── manifest.json
│   ├── shard_0000/
│   └── ...
├── split.json
└── energy_stats.json
```

---

## 七、预处理质量分析

### 文件：`scripts/analyze_preprocessing.py`

对 packed_stage3 中的样本进行质量检验，输出统计报告与可视化。

#### 11 项统计指标

| 指标 | 期望值 |
|------|--------|
| `node_coverage` | > 0.99 |
| `avg_membership_per_node` | 1–5 |
| `max_weight_error` (`|Σw − 1|`) | < 1e-4 |
| `A`（chart 总数） | 10–200 |
| `charts_per_level`（各层） | 均匀分布 |
| `anchor_util` | > 0.5 |
| `overlap_jaccard_mean` | 0.1–0.5 |
| `inter_nn_dist_mean` | < 1.5 Å |
| `inter_normal_dev_mean` | < 0.4 |
| `E_inter` | > 0（当层数 ≥ 2 时） |
| `max_weight_error` | < 1e-4 |

#### 6 项 Pass/Fail 检查

| 检查项 | 阈值 |
|--------|------|
| `node_coverage` | > 0.99 |
| `weight_norm` | `max_weight_error` < 1e-4 |
| `level_diff_valid` | ∈ {−1, +1} |
| `edge_index_range` | `[0, A)` |
| `no_nan_edge_attr` | 无 NaN |
| `inter_edges_exist` | 层数 ≥ 2 时 E_inter > 0 |

#### 输出文件

```
out_dir/
├── summary_stats.json       # mean/std/min/max/p5/p95（所有指标）
├── per_mol_stats.csv        # 每行一个分子
├── flagged_mols.json        # Pass/Fail 失败的分子列表
└── plots/
    ├── chart_count.png
    ├── chart_count_by_level.png
    ├── inter_nn_dist.png
    ├── inter_normal_dev.png
    ├── membership_per_node.png
    ├── overlap_jaccard.png
    ├── anchor_utilization.png
    ├── node_coverage.png
    ├── weight_error.png
    ├── chart_size_by_level.png
    └── mol_<id>_chart_graph.html   # 交互式 3D 图（plotly）
```

```bash
python scripts/analyze_preprocessing.py \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --out-dir    data/ed_energy_5w/preprocess_analysis \
    --n-mols     1000 \
    --plot-mols  308 42 100

# 快速检验（只跑 Pass/Fail，不画图）
python scripts/analyze_preprocessing.py \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --quick-check --n-mols 200
```

---

## 八、训练

### 文件：`scripts/train.py`

支持单卡与多卡 DDP 两种启动方式，自动检测 `LOCAL_RANK` 环境变量。

#### 单卡训练

```bash
python scripts/train.py \
    --data-dir  data/ed_energy_5w \
    --csv-path  data/ed_energy_5w/raw/ed_energy_5w.csv \
    --out-dir   runs/stage6_k3 \
    --num-bblocks 3 --batch-size 8 --epochs 100 --device cuda
```

#### 多卡 DDP（torchrun）

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --data-dir  data/ed_energy_5w \
    --csv-path  data/ed_energy_5w/raw/ed_energy_5w.csv \
    --out-dir   runs/stage6_ddp \
    --num-bblocks 3 --batch-size 8 --grad-accum 2 \
    --epochs 100 --num-workers 4

# 等效 batch = batch_size × num_gpus × grad_accum = 8×8×2 = 128
```

#### 断点续训

```bash
torchrun --nproc_per_node=8 scripts/train.py ... --resume runs/stage6_ddp/last.pt
```

#### 启用 W&B 监控

```bash
torchrun ... \
    --wandb --wandb-project ed2e --wandb-run-name ddp_k3_run1 \
    --wandb-entity <your-entity>
```

#### 关键设计

| 方面 | 设置 |
|------|------|
| 数据集划分 | CSV `scaffold_split` 列（EDBench 原始划分） |
| 能量归一化 | 从 `energy_stats.json` 加载，预处理期间由训练集计算 |
| 损失函数 | 各目标 Huber loss（δ=1.0）取均值 |
| 优化器 | AdamW（lr=1e-3，weight_decay=1e-4） |
| 调度器 | CosineAnnealingLR（T_max=epochs） |
| 验证指标 | per-target MAE（反归一化，单位 Hartree） |
| **精度** | **bf16 AMP（默认开启），H200/A100 推荐；`--no-amp` 可禁用** |
| **梯度累积** | **`--grad-accum N`，等效放大逻辑 batch** |
| **显存优化** | **Gradient Checkpointing（BBlockConfig 默认开启）** |
| **分布式** | **DDP + NCCL，torchrun 自动检测多卡** |
| **数据加载** | **pin_memory=True + non_blocking H2D 传输；persistent_workers** |
| **矩阵乘** | **TF32 默认启用（H200/Ampere+）** |

#### 新增 CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--grad-accum N` | `1` | 梯度累积步数；等效 batch = batch_size × GPUs × N |
| `--no-amp` | — | 禁用 bf16 AMP（默认开启） |
| `--num-workers` | `4` | 每进程 DataLoader workers |

#### W&B 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--wandb` | — | 启用 Weights & Biases 日志（需安装 `pip install wandb`） |
| `--wandb-project` | `ed2e` | W&B 项目名 |
| `--wandb-entity` | — | W&B 团队/用户名，留空使用默认 |
| `--wandb-run-name` | `out-dir` 末段 | Run 名称 |
| `--wandb-tags` | — | 空格分隔的标签列表 |

W&B 额外记录：`grad_accum`、`effective_batch`、`amp`、`grad_checkpointing`、`world_size`。

#### DDP 注意事项

- Checkpoint 和日志只在 **rank 0** 进程执行
- Validation 只在 **rank 0** 进程上运行（避免重复评估）
- `DistributedSampler.set_epoch(epoch)` 在每个 epoch 开始时调用，保证各卡 shuffle 不同
- `model.no_sync()` 在梯度累积的非最后一步跳过 allreduce，节省通信开销

#### Checkpoint 格式

```python
{
    "config":     ED2EConfig,          # dataclass
    "state_dict": model.state_dict(),  # raw_model（非 DDP 包装）的权重
    "norm_mean":  np.ndarray (6,),     # 训练集均值
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

## 九、Attention 可视化

### 文件：`scripts/visualize_attention.py`

```bash
python scripts/visualize_attention.py \
    --checkpoint runs/stage6_k3/best.pt \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --mol-ids 308 42 100 \
    --out-dir  figs/attention
```

#### 输出

| 文件 | 内容 |
|------|------|
| `mol_<id>_<target>.html` | 交互式 plotly 3D scatter，颜色 = 对 head 平均后的 attention 权重 |
| `attention_data.csv` | `mol_id, chart_idx, level_id, x, y, z, attn_t0…t5` |
| `attn_by_level.png` | attention 权重 vs 层级（各目标箱线图，matplotlib） |

**可视化解读**：颜色越亮的 chart 对该能量目标的预测贡献越大。结合 `chart_level_id` 可分析不同密度阈值层对各能量分量的重要性。

---

## 十、冒烟测试

### 文件：`scripts/smoke_stage6_forward.py`

```bash
python scripts/smoke_stage6_forward.py \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --n-mols 4 --device cpu --num-bblocks 1
```

验证项：

| 检查 | 期望 |
|------|------|
| `energy_pred.shape` | `(B, 6)` |
| 无 NaN/Inf | ✓ |
| `attn_weights.shape` | `(T, H, A)` |
| 每分子每目标每 head 注意力之和 | `1.0 ± 1e-4` |
| `chart_level_id.shape` | `(A,)` |
| `chart_center.shape` | `(A, 3)` |

---

## 十一、完整验证命令序列

```bash
# 0. 端到端预处理（首次运行，全量约数小时）
python scripts/preprocess_edbench.py \
    --pkl      data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv      data/ed_energy_5w/raw/ed_energy_5w.csv \
    --data-dir data/ed_energy_5w --workers 8

# 1. 预处理质量分析
python scripts/analyze_preprocessing.py \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --out-dir    data/ed_energy_5w/preprocess_analysis \
    --n-mols 1000 --plot-mols 308 42 100

# 2. Stage 6 冒烟测试
python scripts/smoke_stage6_forward.py \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --n-mols 4 --device cpu --num-bblocks 1

# 3. 训练（单卡）
python scripts/train.py \
    --data-dir data/ed_energy_5w \
    --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \
    --out-dir  runs/stage6_k3 \
    --num-bblocks 3 --batch-size 8 --epochs 100 --device cuda

# 3b. 训练（8× H200 DDP）
torchrun --nproc_per_node=8 scripts/train.py \
    --data-dir data/ed_energy_5w \
    --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \
    --out-dir  runs/stage6_ddp \
    --num-bblocks 3 --batch-size 8 --grad-accum 2 \
    --epochs 100 --num-workers 4

# 4. Attention 可视化
python scripts/visualize_attention.py \
    --checkpoint runs/stage6_k3/best.pt \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --mol-ids 308 42 100 --out-dir figs/attention
```

---

## 十三、数值稳定性

物理特征（∇_M H、‖∇ρ‖ 等）对大分子可达 1e7+ 量级，直接进入线性层 + LayerNorm 会导致方差坍塌 → NaN。当前已在以下位置插入 per-sample 最大值归一化：

| 位置 | 张量 | 归一化方式 |
|------|------|-----------|
| `bblock.py` `_initialize_shared_state` | `node_scalar_raw` (N,5) | `/ amax(dim=-1).clamp_min(1)` |
| `bblock.py` `_initialize_shared_state` | `node_vector_raw` (N,2,3) | `/ amax(dim=(-2,-1)).clamp_min(1)` |
| `stage3_local.py` `encode_structure` | `chart_es_geom_static` (A,53) | `/ amax(dim=-1).clamp_min(1)` |
| `stage3_local.py` `LocalMessagePassingLayer` | `local_edge_attr` (E,6) | `/ amax(dim=-1).clamp_min(1)` |
| `stage4_intra.py` `IntraLevelBlock` | `intra_geom_static` (A,7) | `/ amax(dim=-1).clamp_min(1)` |
| `stage5_inter.py` `InterLevelBlock` | `inter_level_edge_attr` (E,7) | `/ amax(dim=-1).clamp_min(1)` |

所有 `.norm(dim=-1)` 调用（12 处，分布于 stage3/4/5 和 readout）已替换为 `_safe_norm()`（在 sqrt 内部加 ε，防止零模长处梯度为 NaN）：

```python
def _safe_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return (x.pow(2).sum(dim=dim) + 1e-8).sqrt()
```

训练脚本还额外提供：
- 输入批次 NaN/Inf 检测（跳过问题批次）
- loss NaN/Inf 检测（跳过梯度步）
- 梯度 NaN/Inf 检测（跳过 optimizer.step）

诊断脚本：

```bash
python scripts/debug_nan_forward.py \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --mol-ids 1370746 2352995 --device cpu --num-bblocks 1
```

默认配置（K=3，H=4，token_dim=96）：

| 组件 | 参数量（约） |
|------|------------|
| FCLCLocalBlock × 3 | ~1.3M |
| IntraLevelBlock × 3 | ~0.7M |
| InterLevelBlock × 3 | ~0.5M |
| MultiHeadChartReadout | ~0.1M |
| EnergyHeads × 6 | ~0.07M |
| **合计** | **~2.7M** |
