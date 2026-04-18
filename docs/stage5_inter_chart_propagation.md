# Stage 5: 层间 Chart 图消息传递（T_inter）

## 当前范围

本阶段实现的是 **Stage 5 T_inter block**，即在相邻层 chart 图之间做双向消息传递。

已落地内容：

- Stage 2 数据层扩展（`FCLCLevel.inter_weights` 4 元组 + `inter_weights_up`）
- Stage 3 数据层扩展（`inter_level_edge_index`、`inter_level_weights`、`inter_level_edge_attr`）
- `Stage3PackedDataset`：`inter_edge` count key + manifest 分片支持
- `Stage3ShardedWriter`：流式写入，自动分片
- `InterLevelBlock` 完整 forward
- 统一预处理脚本 `preprocess_stage2_to_packed.py`
- 冒烟测试脚本

当前明确**不做**：

- 能量读出头
- loss / 反传

---

## 在 B-Block 中的位置

```
T_inter ∘ T_intra ∘ T_chart_encode ∘ (T_local_msg)³ ∘ T_init
↑ 本文档
```

Stage 5 消费 Stage 4 输出的 `p_bar: DualStreamState`（`(A, 64)` scalar + `(A, 8, 2)` vector），
返回更新后的 `DualStreamState`。

---

## 传播方向与边权重约定

**双向**，对每对相邻层 (k, k+1) 分别计算两个方向：

| 方向 | 数据来源字段 | 顶点投影 |
|------|------------|---------|
| k ← k+1（外层接收） | `fclc_levels[k].inter_weights` | `proj = verts_k + 0.5 * normals_k` |
| k+1 ← k（内层接收） | `fclc_levels[k+1].inter_weights_up` | `proj = verts_k1 - 0.5 * normals_k1` |

`level_id=0` = 最外层（最低密度阈值）：

- k ← k+1 方向：`level_diff = recv_level − send_level = −1`
- k+1 ← k 方向：`level_diff = +1`

两个方向合并为单一 `inter_level_edge_index`，共 `E_inter` 条边。

---

## 新增数据字段

### `FCLCLevel` 扩展

| 字段 | 类型 | 含义 |
|------|------|------|
| `inter_weights` | `Dict[int, List[Tuple[int, float, float, float]]]` | 4 元组 `(chart_id_b, w̃, mean_nn_dist, mean_normal_dev)` |
| `inter_weights_up` | 同上 | 反向 k+1 ← k（存储在 level k+1 上） |

旧 2 元组 Stage 2 cache 加载时自动抛出 `RuntimeError`，须重建。

### `Stage3Sample` / `Stage3TensorBatch`

| 字段 | 类型 | 形状 | 含义 |
|------|------|------|------|
| `inter_level_edge_index` | int64 | `(2, E_inter)` | 层间 chart 边（send→recv） |
| `inter_level_weights` | float32 | `(E_inter,)` | w̃_{recv←send} counting-based 权重 |
| `inter_level_edge_attr` | float32 | `(E_inter, 7)` | 7 维层间几何特征 |

### `inter_level_edge_attr` 7 维含义

| 索引 | 符号 | 来源 |
|------|------|------|
| 0 | `mean_nn_dist` | 投影顶点到目标层 NN 的平均距离（精确，Stage 2 计算） |
| 1 | `mean_normal_dev` | `1 − cos(n_src, n_target_nn)` 均值（精确，Stage 2 计算） |
| 2 | `center_dist` | `‖c_recv − c_send‖` 3D 中心距离 |
| 3 | `cos_theta` | `f_recv[0] · f_send[0]` frame 主轴余弦 |
| 4 | `sin_theta` | `f_recv[1] · f_send[0]` frame 主轴正弦 |
| 5 | `log_size_ratio` | `log((|P_recv|+1)/(|P_send|+1))` |
| 6 | `level_diff` | `recv_level − send_level` ∈ {−1, +1} |

### Collate offset 规则

| 字段 | Offset 规则 |
|------|------------|
| `inter_level_edge_index` | 两行均 `+= chart_offset` |
| `inter_level_weights` | 无 offset |
| `inter_level_edge_attr` | 无 offset |

### Cache tag bump

旧格式（Stage 4）：`stage3_lk{k}_ck{k}_a{n}_ig7`

新格式（含 Stage 5 字段）：`stage3_lk{k}_ck{k}_a{n}_ig7_il7`

旧 cache 自动失效，须重新运行 `preprocess_stage3.py` 或 `preprocess_stage2_to_packed.py`。

---

## `InterLevelBlock` 子网络表

| 子网络 | 类型 | 输入→隐→输出 | 作用 |
|--------|------|------------|------|
| `enc_e` | `_MLP` | 7→96→96 | e_ab → F_e（边特征编码） |
| `psi_inter` | `_MLP` | 224→96→1 | `[p̄_dst^s, p̄_src^s, F_e]` → content logit |
| `mlp_beta` | `_MLP` | 352→96→1 | z_ab → structural bias β |
| `mlp_gate_s` | `_MLP` | 352→96→64 | z_ab → scalar gate g_s |
| `mlp_gate_v` | `_MLP` | 352→96→8 | z_ab → vector gate g_v |
| `phi_s` | `_MLP` | 64→96→64 | sender scalar → message seed |
| `phi_v` | `_MLP` | 16→96→16 | sender vector（展平 8×2）→ message seed |
| `mlp_update_s` | `_MLP` | 136→96→64 | `[p̄^s, m_a^s, ‖m_a^v‖]` → delta_s |
| `scalar_update_norm` | LayerNorm | 64 | 残差后归一化 |
| `mlp_vgate` | `_MLP` | 128→96→8 | `[p̄^s, m_a^s]` → vector gate |
| `scalar_to_v` | `_MLP` | 64→96→16 | p_new^s → 8×2 |

**z_ab**（352 维）= `[p̄_dst^s, p̄_src^s, F_e, p̄_dst^s − p̄_src^s, p̄_dst^s ⊙ p̄_src^s]`
= 64 + 64 + 96 + 64 + 64 = 352

---

## InterLevelBlock 完整 Forward 流程

**Step 1 — 空图判断**

```python
if inter_static["inter_level_edge_index"].numel() == 0:
    return p_bar
```

**Step 2 — 边特征编码**

```python
F_e = enc_e(e_ab)                                      # (E, 96)
z_ab = cat([p̄_dst^s, p̄_src^s, F_e, p̄_dst^s - p̄_src^s, p̄_dst^s * p̄_src^s])  # (E, 352)
```

**Step 3 — 注意力**

```python
beta  = mlp_beta(z_ab).squeeze(-1)
g_s   = sigmoid(mlp_gate_s(z_ab))
g_v   = sigmoid(mlp_gate_v(z_ab))
logit = psi_inter(cat([p̄_dst^s, p̄_src^s, F_e])).squeeze(-1)
log_w = log(inter_level_weights + ε)
alpha = _segment_softmax(logit + log_w + beta, dst, A)
```

**Step 4 — 消息**

```python
m_s = alpha[:,None] * g_s * phi_s(p̄_src^s)
m_v = alpha[:,None,None] * g_v[:,None] * phi_v(p̄_src^v.flatten(1)).view(-1, 8, 2)
```

**Step 5 — 聚合**

```python
m_a_s = _scatter_add(m_s, dst, A)    # (A, 64)
m_a_v = _scatter_add(m_v, dst, A)    # (A, 8, 2)
```

**Step 6 — 更新**

```python
delta_s  = mlp_update_s(cat([p̄^s, m_a^s, ‖m_a^v‖]))
p_new_s  = scalar_update_norm(p̄^s + delta_s)
gate_v   = sigmoid(mlp_vgate(cat([p̄^s, m_a^s]))).unsqueeze(-1)
from_s   = scalar_to_v(p_new_s).view(A, 8, 2)
p_new_v  = p̄^v + gate_v * m_a^v + 0.1 * from_s

return DualStreamState(scalar=p_new_s, vector=p_new_v)
```

---

## 与 Stage 4 的关键设计对比

| 方面 | Stage 4 (T_intra) | Stage 5 (T_inter) |
|------|------------------|------------------|
| 传播方向 | 同层单向（chart_graph） | 双向跨层（k↔k+1） |
| 边权重 | 无（纯注意力） | w̃_{a←b} 作为 log-bias |
| 边特征 | 无（用 intra_geom 编码 chart 身份） | 7 维，精确顶点投影统计 |
| 边特征编码 | ExplicitStructureEncoderIntra（3-token MHA） | enc_e: _MLP(7→96→96) |
| z_ab 维度 | 384 | 352 |
| Overlap 修正 | 有 OverlapContextEncoder | 无 |
| 注意力 logit | `logit + β` | `logit + log_w + β` |

---

## Stage3PackedDataset 与 Stage3ShardedWriter

### Stage3ShardedWriter（流式写入）

```python
writer = Stage3ShardedWriter(root="data/packed_stage3", shard_size=2000)
for mol_id in mol_ids:
    sample = build_stage3_sample(...)
    writer.put(sample)
writer.finalize()
# 输出: data/packed_stage3/manifest.json + shard_0000/ + shard_0001/ + ...
```

### Stage3PackedDataset（manifest 分片读取）

```python
dataset = Stage3PackedDataset("data/packed_stage3")  # 自动检测 manifest.json
# 等价于合并所有 shard 的单一视图
```

---

## CLI

### 统一预处理（Stage 2 + 3 + 5 prep 一次完成）

```bash
python scripts/preprocess_stage2_to_packed.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --max-samples 50
```

### 或保留分步方式（Stage 3 重建含 Stage 5 字段）

```bash
python scripts/preprocess_stage3.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --fclc-source data/ed_energy_5w/cache_fclc \
    --cache-dir data/ed_energy_5w/cache_stage3 \
    --max-samples 50
```

注：Stage 2 `cache_fclc` 若为旧 2 元组格式，加载时会自动抛出 `RuntimeError`，须先重建 Stage 2 cache：

```bash
python scripts/preprocess_stage2.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --cache-dir data/ed_energy_5w/cache_fclc \
    --max-samples 50
```

### Stage 5 冒烟测试

```bash
python scripts/smoke_stage5_inter_forward.py \
    --cache-dir data/ed_energy_5w/cache_stage3 \
    --n-mols 4 --device cpu

# 或从 packed 格式读取
python scripts/smoke_stage5_inter_forward.py \
    --packed-dir data/ed_energy_5w/packed_stage3 \
    --n-mols 4 --device cpu
```

---

## 数据流检查项

| 检查项 | 期望 |
|--------|------|
| `batch.inter_level_edge_index.shape` | `(2, E_inter)` |
| `batch.inter_level_weights.shape` | `(E_inter,)` |
| `batch.inter_level_weights` 值域 | `[0, 1]` |
| `batch.inter_level_edge_attr.shape` | `(E_inter, 7)` |
| `level_diff` 值域 | `{−1, +1}` |
| src/dst 范围 | `[0, A)` |
| Stage 5 forward 无 NaN/Inf | ✓ |
| 输出 scalar shape | `(A, 64)` |
| 输出 vector shape | `(A, 8, 2)` |
