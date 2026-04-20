# Stage 4: 层内 Chart 图消息传递（T_intra）

## 当前范围

本阶段实现的是 **Stage 4 T_intra block**，即在同层 chart 图上做一轮 chart 间消息传递。

已落地内容：

- Stage 3 数据层扩展（`intra_geom_static`、`overlap_edge_to_chart_edge_index`、`chart_to_ref`）
- `IntraLevelBlock` 完整 forward（含显式结构编码 + overlap-context 修正）
- 冒烟测试脚本

当前明确**不做**：

- 跨层传播（Stage 5 T_inter）
- 能量读出头
- loss / 反传

---

## 在 B-Block 中的位置

```
T_inter ∘ T_intra ∘ (T_chart_encode ∘ T_local_msg) × num_local_steps ∘ T_chart_encode ∘ T_init
                ↑ 本文档
```

Stage 3 (`FCLCLocalBlock`) 输出：
- `p_next_local: DualStreamState` — 每 chart 的状态 `(A, 64)` scalar + `(A, 8, 2)` vector
- `local_state_final: DualStreamState` — 每 membership entry 的局部状态 `(N_M, 64)` + `(N_M, 8, 2)`
- `intra_static_bundle` — 静态 chart 图拓扑

Stage 4 消费上述输出，返回更新后的 `DualStreamState`（`p_bar`），交给 Stage 5。

---

## 三层状态与 Stage 4 操作层

| 层级 | 实体 | 状态 | 操作阶段 |
|------|------|------|----------|
| 共享节点 | N 个网格节点 | `(N, 64)` + `(N, 8, 2)` | Stage 3 写入 |
| Membership | N_M 个 chart×node 对 | `(N_M, 64)` + `(N_M, 8, 2)` | Stage 3 写入；Stage 4 只读 |
| **Chart** | **A 个 chart** | **(A, 64) + (A, 8, 2)** | **Stage 4 更新** |

Stage 4 操作的是 chart 层；共享节点和 membership 状态只被 `OverlapContextEncoder` 只读使用。

---

## 新增数据字段（Stage 4）

### `Stage3Sample` / `Stage3TensorBatch`

| 字段 | 类型 | 形状 | 含义 |
|------|------|------|------|
| `intra_geom_static` | float32 | `(A, 7)` | 每个 chart 相对其组参考 chart 的静态几何描述 |
| `overlap_edge_to_chart_edge_index` | int64 | `(E_ov,)` | 每条 overlap 边在 `chart_graph_edge_index` 中的位置 |
| `chart_frame_metadata["chart_to_ref"]` | int64 | `(A,)` | 每个 chart 对应其组参考 chart 的全局 chart 索引 |

> **`chart_to_ref`（A,）与 `reference_chart_id`（G,）的关系**：
> - `reference_chart_id[g]`：第 g 组的 medoid 的全局 chart 索引，共 G 个（每 level×component 一组）
> - `chart_to_ref[a]`：chart a 所属组的 medoid 的全局 chart 索引，共 A 个
>
> 两者的值域相同（均为全局 chart 索引），但形状不同：`chart_to_ref` 是对每个 chart 的广播展开，
> 而 `reference_chart_id` 是去重后的组级列表。等价关系：
> `set(chart_to_ref.tolist()) == set(reference_chart_id.tolist())`

### `intra_geom_static` 7 维含义

`[d_M, Δx_u, Δx_v, 1-n·n₀, cosθ, sinθ, log_area_ratio]`

| 索引 | 符号 | 含义 |
|------|------|------|
| 0 | `d_M` | chart a → 组参考 chart 的测地线距离（seed-to-seed Dijkstra） |
| 1,2 | `Δx_u, Δx_v` | chart a 中心在参考 chart 坐标系下的 2D 投影差（= π_a - π₀） |
| 3 | `1-n·n₀` | 法向偏差（= 0 表示平行） |
| 4,5 | `cosθ, sinθ` | chart a 的 frame 主轴在参考 frame 下的分量（frame 旋转编码） |
| 6 | `log_area_ratio` | `log((size_a+1)/(size_ref+1))` 大小比对数 |

### Collate offset 规则

| 字段 | Offset 规则 |
|------|------------|
| `intra_geom_static` | 无 offset（直接 concat） |
| `overlap_edge_to_chart_edge_index` | 先 append，后 `+= sample.chart_graph_edge_index.shape[1]` |
| `chart_frame_metadata["chart_to_ref"]` | `+ chart_offset`（chart 全局索引） |

### Cache tag

旧格式（无新字段）：`stage3_lk{k}_ck{k}_a{n}`

新格式（含新字段）：`stage3_lk{k}_ck{k}_a{n}_ig7`

旧 cache 自动失效，须重新运行 `preprocess_stage3.py`。

### Packed cache（`stage3_packed.py`）

新增字段写入：`intra_geom_static.npy`（count_key="chart"）、`overlap_edge_to_chart_edge_index.npy`（count_key="overlap_edge"）；`chart_to_ref` 加入 `_META_CHART_FIELDS` 由现有框架自动处理。

---

## 子网络表

### `ExplicitStructureEncoderIntra`

与 Stage 3 `ExplicitStructureEncoder` 结构相同（3-token typed attention），但输入维度不同，且只输出单路 `F̃_a`（不需要 vector_seed 和双路输出头）。

| 子网络 | 类型 | 输入→隐→输出 | 作用 |
|--------|------|------------|------|
| `enc_g` | `_MLP` | 7→96→96 | geom_static → token |
| `enc_s` | `_MLP` | 192→96→96 | 3×64 scalar ES → token |
| `enc_v` | `_MLP` | 72→96→96 | 9×8 vector ES → token |
| `type_embedding` | Parameter | (3, 96) | 三类 token 类型嵌入 |
| `attn` | MultiheadAttention | dim=96, heads=4 | 3-token 自注意力 |
| `token_norm` | LayerNorm | 96 | 注意力后 residual + norm |
| `fuse_norm` | LayerNorm | 96 | mean pool 后归一化 |
| `F_tilde_head` | `_MLP` | 96→96→96 | pooled → F̃_a |

**scalar ES（192 维）**：`[p_a^s, p_ref^s, p_a^s - p_ref^s]`

**vector ES（72 维）**：`[p_a_in_ref(16), p_ref_in_ref(16), diff(16), ‖p_a^v‖(8), ‖p_ref^v‖(8), dot(8)]`

向量需先旋转到参考 chart frame：`_rotate_vectors(p.vector, chart_frame, chart_frame[ref_idx])`

### `OverlapContextEncoder`

| 子网络 | 类型 | 输入→隐→输出 | 作用 |
|--------|------|------------|------|
| `mlp_ctx` | `_MLP` | 73→96→64 | (64+8+1) → C_ab |

**C_ab 构造**：对每条 overlap 边，从 CSR `overlap_shared_ptr` 展开 segment index（`torch.repeat_interleave`），对共享 membership 的 scalar/vector norm 做 `_scatter_mean`，拼接 jaccard 系数后过 MLP。

### `IntraLevelBlock`

#### 边调制与注意力

| 子网络 | 类型 | 输入→隐→输出 | 作用 |
|--------|------|------------|------|
| `psi_intra` | `_MLP` | 128→96→1 | `[p_dst^s, p_src^s]` → content logit |
| `mlp_beta` | `_MLP` | 384→96→1 | z_ab → structural bias β |
| `mlp_gate_s` | `_MLP` | 384→96→64 | z_ab → scalar gate g_s |
| `mlp_gate_v` | `_MLP` | 384→96→8 | z_ab → vector gate g_v |

z_ab = `[F̃_dst, F̃_src, F̃_dst − F̃_src, F̃_dst ⊙ F̃_src]`，dim = 4×96 = 384

#### 发送端变换

| 子网络 | 类型 | 输入→隐→输出 | 作用 |
|--------|------|------------|------|
| `phi_s` | `_MLP` | 64→96→64 | sender scalar → message seed |
| `phi_v` | `_MLP` | 16→96→16 | sender vector（展平 8×2）→ message seed |

#### Overlap-context 修正

| 子网络 | 类型 | 输入→隐→输出 | 作用 |
|--------|------|------------|------|
| `psi_ctx_s` | `_MLP` | 129→96→64 | `[m_s[ov2ce], C_ab, jaccard]` → Δm_s |

修正以 autograd-safe 方式写入：在全零张量上 `scatter_add_`，再与 `m_s` 相加（不做 in-place）。

#### Chart 状态更新

| 子网络 | 类型 | 输入→隐→输出 | 作用 |
|--------|------|------------|------|
| `mlp_update_s` | `_MLP` | 136→96→64 | `[p^s, m_a^s, ‖m_a^v‖]` → scalar delta |
| `scalar_update_norm` | LayerNorm | 64 | 残差后归一化 |
| `mlp_vgate` | `_MLP` | 128→96→8 | `[p^s, m_a^s]` → vector gate |
| `scalar_to_v` | `_MLP` | 64→96→16 | p̄^s → 8×2 |

---

## IntraLevelBlock 完整 Forward 流程

```python
def forward(p_mid, local_state_final, batch, intra_static) -> DualStreamState:
```

**Step 1 — 空图判断**

```python
if intra_static["chart_graph_edge_index"].numel() == 0:
    return p_mid
```

**Step 2 — 构建 ES，编码 F̃_a**

```python
ref_idx   = intra_static["chart_to_ref"]
p_ref_s   = p_mid.scalar[ref_idx]
scalar_es = cat([p_mid.scalar, p_ref_s, p_mid.scalar - p_ref_s])  # (A, 192)

chart_frame   = batch.chart_frame_metadata["chart_frame"]
p_a_in_ref    = _rotate_vectors(p_mid.vector, chart_frame, chart_frame[ref_idx])
p_ref_in_ref  = p_mid.vector[ref_idx]
diff_in_ref   = p_a_in_ref - p_ref_in_ref
vec_es = cat([p_a_in_ref.flatten(1), p_ref_in_ref.flatten(1), diff_in_ref.flatten(1),
              _safe_norm(p_mid.vector),            # 替代 .norm(-1)
              _safe_norm(p_mid.vector[ref_idx]),
              (p_a_in_ref * p_ref_in_ref).sum(-1)])   # (A, 72)

# intra_geom_static per-chart 最大值归一化，防止极端物理量（e.g. 大分子 d_M）导致 LayerNorm 方差坍塌
geom_intra = intra_static["intra_geom_static"]
g_scale    = geom_intra.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
F̃ = es_encoder(geom_intra / g_scale, scalar_es, vec_es)  # (A, 96)
```

**Step 3 — 主消息**

```python
src, dst = chart_graph_edge_index
z_ab = cat([F̃[dst], F̃[src], F̃[dst]-F̃[src], F̃[dst]*F̃[src]])  # (E, 384)
beta = mlp_beta(z_ab).squeeze(-1)
g_s  = sigmoid(mlp_gate_s(z_ab))
g_v  = sigmoid(mlp_gate_v(z_ab))

logit = psi_intra(cat([p_mid.scalar[dst], p_mid.scalar[src]])).squeeze(-1)
alpha = _segment_softmax(logit + beta, dst, A)

m_s = alpha[:,None] * g_s * phi_s(p_mid.scalar[src])             # (E, 64)
m_v = alpha[:,None,None] * g_v[:,None] * phi_v(p_mid.vector[src].flatten(1)).view(-1, 8, 2)
```

**Step 4 — Overlap-context 修正**

```python
if E_ov > 0:
    C_ab  = overlap_ctx_enc(local_state_final, intra_static)      # (E_ov, 64)
    delta = psi_ctx_s(cat([m_s[ov2ce], C_ab, jaccard[:,None]]))   # (E_ov, 64)
    delta_full = zeros_like(m_s)
    delta_full.scatter_add_(0, ov2ce[:,None].expand_as(delta), delta)
    m_s = m_s + delta_full                                         # (E, 64)  autograd-safe
```

**Step 5 — 聚合**

```python
m_a_s = _scatter_add(m_s, dst, A)   # (A, 64)
m_a_v = _scatter_add(m_v, dst, A)   # (A, 8, 2)
```

**Step 6 — 更新**

```python
delta_s = mlp_update_s(cat([p_mid.scalar, m_a_s, _safe_norm(m_a_v)]))
p_bar_s = scalar_update_norm(p_mid.scalar + delta_s)

gate_v  = sigmoid(mlp_vgate(cat([p_mid.scalar, m_a_s]))).unsqueeze(-1)
from_s  = scalar_to_v(p_bar_s).view(A, 8, 2)
p_bar_v = p_mid.vector + gate_v * m_a_v + 0.1 * from_s

return DualStreamState(scalar=p_bar_s, vector=p_bar_v)
```

---

## Stage 4 输出接口 / Stage 5 所需数据

### Stage 4 输出

| 名称 | 形状 | 含义 |
|------|------|------|
| `p_bar.scalar` | `(A, 64)` | 更新后 chart scalar（chart 自身坐标系） |
| `p_bar.vector` | `(A, 8, 2)` | 更新后 chart vector（chart 自身坐标系） |

调用方保持 `batch: Stage3TensorBatch` 和 `local_state_final` 不变，整体传入 Stage 5。

### Stage 5 (T_inter) 所需额外数据（**当前缺失，待 Stage 5 实现时添加**）

| 字段 | 形状 | 来源 |
|------|------|------|
| `inter_level_edge_index` | `(2, E_inter)` | Stage 2 `compute_inter_layer_weights()` 结果，Stage 5 预处理时写入 `Stage3Sample` |
| `inter_level_weights` | `(E_inter,)` | 同上，w̃_{a←b} counting-based 权重 |

`chart_frame_metadata` 中的 `chart_level_id`、`chart_component_id`、`chart_group_id` 已包含层间分组信息，Stage 5 可直接使用。Stage 5 实现时须再次 bump cache tag。

---

## CLI

### 重建 Stage 3 cache（包含新字段）

```bash
python scripts/preprocess_stage3.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --fclc-source data/ed_energy_5w/cache_fclc \
    --cache-dir data/ed_energy_5w/cache_stage3 \
    --max-samples 50
```

### Stage 4 冒烟测试

```bash
python scripts/smoke_stage4_intra_forward.py \
    --cache-dir data/ed_energy_5w/cache_stage3 \
    --n-mols 4 --device cpu
```

---

## 数据流检查项

| 检查项 | 期望 |
|--------|------|
| `batch.intra_geom_static.shape` | `(A, 7)` |
| `batch.overlap_edge_to_chart_edge_index.shape` | `(E_ov,)` |
| `ov2ce` 值范围 | `[0, E_chart)` |
| `chart_to_ref` 值范围 | `[0, A)` |
| Stage 4 forward 无 NaN/Inf | ✓ |
| 输出 scalar shape | `(A, 64)` |
| 输出 vector shape | `(A, 8, 2)` |

---

## 当前 Smoke 结果

4 个分子，CPU，`all_stage3_lk12_ck8_a8.zip`：

```
Loaded 4 molecules
  A=7778  E_chart=63539  E_overlap=10932
  intra_geom_static  (7778, 7)  ✓
  ov2ce              (10932,)  range [0, 63539)  ✓
  chart_to_ref       (7778,)  range [0, 7778)  ✓

Stage4 forward OK
  scalar (7778, 64)  vector (7778, 8, 2)  no NaN ✓
```
