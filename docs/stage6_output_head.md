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
[T_inter ∘ T_intra ∘ T_chart_encode ∘ (T_local_msg)³ ∘ T_init] × K
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
batch (Stage3TensorBatch)
  → FCLCLocalBlock(batch, shared_state=shared_state)
      → node_state_shared_next (N, 64+8×2)   ← 传给下一轮 T_init
      → local_state_final      (N_M, ...)     ← 节点状态，供 overlap ctx
      → p_next_local           (A, 64+8×2)   ← chart 状态（T_chart_encode 输出）
  → IntraLevelBlock(p_next_local, local_state_final, batch, intra_static)
      → p_bar  (A, 64+8×2)
  → InterLevelBlock(p_bar, batch, inter_static)
      → p_new  (A, 64+8×2)
```

`intra_static` 和 `inter_static` 在 `ED2EBBlockStack.forward()` 中各预计算一次，在所有 K 轮 B-block 之间复用（静态几何，不随迭代改变）。

#### `shared_state` 传递规则

| 轮次 | 输入 `shared_state` | 输出 `node_state_shared_next` |
|------|--------------------|-----------------------------|
| k=0 | `None`（FCLCLocalBlock 内部初始化） | (N, 64) + (N, 8, 2) |
| k=1 | 上轮 `node_state_shared_next` | 同上 |
| k=K-1 | 同上 | 丢弃（只用 `p_new`） |

> **注**：`shared_state` 是 per-**NODE**（形状 (N, ...)），FCLCLocalBlock 内部通过 `membership[:, 1]`（node 索引）将其映射到对应 chart。`FCLCLocalBlock.forward()` 中 `shared_state` 是 keyword-only 参数，调用时必须写 `shared_state=...`。

#### Config

```python
@dataclass
class BBlockConfig:
    num_bblocks: int = 3
    local_cfg:   Stage3LocalConfig = field(default_factory=Stage3LocalConfig)
    intra_cfg:   Stage4IntraConfig = field(default_factory=Stage4IntraConfig)
    inter_cfg:   Stage5InterConfig = field(default_factory=Stage5InterConfig)
```

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

```bash
python scripts/train.py \
    --data-dir  data/ed_energy_5w \
    --csv-path  data/ed_energy_5w/raw/ed_energy_5w.csv \
    --out-dir   runs/stage6_k3 \
    --num-bblocks 3 --batch-size 8 --epochs 100 --device cuda

# 断点续训
python scripts/train.py ... --resume runs/stage6_k3/last.pt
```

#### 关键设计

| 方面 | 设置 |
|------|------|
| 数据集划分 | CSV `scaffold_split` 列（EDBench 原始划分） |
| 能量归一化 | 从 `energy_stats.json` 加载，预处理期间由训练集计算 |
| 损失函数 | 各目标 Huber loss（δ=1.0）取均值 |
| 优化器 | AdamW（lr=1e-3，weight_decay=1e-4） |
| 调度器 | CosineAnnealingLR（T_max=epochs） |
| 梯度裁剪 | 不做（初版） |
| 验证指标 | per-target MAE（反归一化，单位 Hartree） |

#### Checkpoint 格式

```python
{
    "config":     ED2EConfig,          # dataclass
    "state_dict": model.state_dict(),
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

# 3. 训练
python scripts/train.py \
    --data-dir data/ed_energy_5w \
    --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \
    --out-dir  runs/stage6_k3 \
    --num-bblocks 3 --batch-size 8 --epochs 100 --device cuda

# 4. Attention 可视化
python scripts/visualize_attention.py \
    --checkpoint runs/stage6_k3/best.pt \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --mol-ids 308 42 100 --out-dir figs/attention
```

---

## 十二、参数规模参考

默认配置（K=3，H=4，token_dim=96）：

| 组件 | 参数量（约） |
|------|------------|
| FCLCLocalBlock × 3 | ~1.3M |
| IntraLevelBlock × 3 | ~0.7M |
| InterLevelBlock × 3 | ~0.5M |
| MultiHeadChartReadout | ~0.1M |
| EnergyHeads × 6 | ~0.07M |
| **合计** | **~2.7M** |
