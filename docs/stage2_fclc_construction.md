# Stage 2: Feature-Compatible Local Chart (FCLC) Atlas Construction

## 概述

本阶段在 Stage 1 输出的每个三角网格连通分量（`ManifoldComponent`）上，构造一组**特征相容局部坐标卡（FCLC）atlas**。

**核心方法：** 以 geodesic medoid 为初始种子点，用区域相容打分（$S_R$）生长第一个 chart；随后以前沿驱动的贪心扩展算法逐步覆盖全部顶点，每次从前沿选出最高质量候选 chart 加入 atlas。每个 chart 通过切平面局部 PCA 构造确定性参考系，划分为四个象限，并计算局部显式结构描述子 $ES_a^{\text{local}}$。最后按法向投影计数方式计算相邻密度层之间的方向性对应权重。

**输入**（来自 Stage 1）：
- `List[ManifoldLevel]`，每层包含若干 `ManifoldComponent`（三角网格 + 顶点特征）

**输出**：
- `List[FCLCLevel]`：每层一个 `FCLCLevel`，含若干 `FCLCChart`
- 每个 `FCLCLevel` 含预计算的层间对应权重 `inter_weights`

---

## 设计原则

| 原则 | 实现方式 |
|---|---|
| **无随机性** | Geodesic medoid 用等步长下采样（`stride = V // 64`）；PCA frame 用 `np.linalg.eigh`（确定性升序特征值） |
| **并行化** | 预处理脚本默认使用单进程多线程；也保留 `fork` 进程池后端供需要时切换 |
| **缓存** | `List[FCLCLevel]` 先按分子序列化为 pkl，最终默认流式打包为单文件 zip bundle；仅在 legacy 模式下才合并为 `{mol_id: List[FCLCLevel]}` dict |
| **高性能** | 中等规模连通分量预计算全局测地距离矩阵（`shortest_path`，一次/连通分量，默认 `V ≤ 3000`）；更大分量退回按需 Dijkstra；前沿计算 Numba `@njit` 编译；`covered_mask` 为 numpy bool 数组 |

---

## 文件位置

```
ed2e/
├── data/
│   └── fclc.py                  ← Stage 2 全部核心代码
└── utils/
    └── visualize_fclc.py        ← FCLC atlas 3-D 可视化
scripts/
├── preprocess_stage2.py         ← 批量并行预处理
└── ablate_fclc_chart_size.py    ← 超参数消融实验
```

---

## 完整流水线

```
List[ManifoldLevel]
    │
    └─ 对每个 ManifoldComponent (V 个顶点, F 个三角面):
        ├─ build_mesh_adjacency()         → 稀疏邻接图 (V×V)
        ├─ shortest_path(adj)             → 全局测地距离矩阵 (V×V)，默认仅 V≤3000 时预计算
        ├─ geodesic_medoid()              → 初始种子点 c_1（等步长下采样，确定性）
        ├─ grow_chart(c_1; dist_mat[c_1]) → 初始 chart P_1
        └─ 前沿驱动 atlas 生长循环 (直到覆盖率 ≥ 99%):
            ├─ _compute_frontier_jit()   → 前沿边缘点集 F_t（Numba @njit，原始 CSR 数组）
            ├─ _frontier_scores_jit()    → S_frontier(b) = λ_U U + λ_D D + λ_C C（Numba @njit）
            ├─ _s2_score()               → 二点参考邻域 N_2(b*)（使用 dist_mat[b*]）
            ├─ grow_chart(e; dist_mat[e])→ 每个候选发起点的候选 chart
            └─ _candidate_quality()      → Q(e) = α_U U_P + α_D D_P + α_C C_P - α_F L_front
    │
    ├─ compute_pca_frame_and_coords()     → frame (2,3), local_coords (V_a,2), quadrant (V_a,)
    ├─ compute_es_local()                 → ES_a^local (53,) float32
    │
    └─ compute_inter_layer_weights()      → 层间计数权重 w̃_{a←b}
```

---

## 数据结构

### `FCLCChart`

一个局部坐标卡 $P_a$。

```python
@dataclass
class FCLCChart:
    chart_id:      int
    level_id:      int
    component_id:  int

    vert_indices:  np.ndarray   # (V_a,) int32 — ManifoldComponent 内的顶点索引

    # 几何（局部 PCA frame）
    center:        np.ndarray   # (3,)     float32 — 中心点坐标（Bohr）
    center_normal: np.ndarray   # (3,)     float32 — 中心点单位法向
    frame:         np.ndarray   # (2, 3)   float32 — (e_{a,1}, e_{a,2})，PCA 切平面正交基
    local_coords:  np.ndarray   # (V_a, 2) float32 — 2D 切平面投影坐标
    quadrant:      np.ndarray   # (V_a,)   int8    — 四象限标签 {0,1,2,3}

    # 来自 Stage 1 的特征
    scalar_feats:  np.ndarray   # (V_a, 5)    float32
    vector_feats:  np.ndarray   # (V_a, 2, 3) float32

    # 局部显式结构描述子（预计算）
    es_local:      np.ndarray   # (53,) float32

    # Stage 3 接口升级
    seed_vertex_idx: int        # 原始 chart 种子点（component 内局部索引）
    membership_sr: Optional[np.ndarray]  # (V_a,) float32，与 vert_indices 对齐
```

### `FCLCLevel`

一个密度层的全部 chart。

```python
@dataclass
class FCLCLevel:
    level_id:   int
    threshold:  float
    charts:     List[FCLCChart]

    # 层间方向性对应权重（本层 k 为接收端，层 k+1 为发送端）
    # inter_weights[chart_id_a] = [(chart_id_b, w̃_{a←b}), ...]
    inter_weights: Optional[Dict[int, List[Tuple[int, float]]]] = None
```

### Stage 3 依赖说明

从当前版本开始，Stage 2 cache 不仅服务于 atlas 构造和层间权重，还会直接被 Stage 3 读取。因此：

- `seed_vertex_idx` 用于在 Stage 3 中恢复 chart 的测地种子与 chart graph 的 geodesic 参考点
- `membership_sr` 用于构造 `membership_weight`

其中 `membership_weight` 不在 Stage 2 中缓存，而是在 Stage 3 中按同一物理节点上的

`exp(-membership_sr)`

归一化得到。

---

## 算法详解

### Geodesic Medoid（确定性）

$$c_1 = \arg\min_{v \in \mathcal{S}} \sum_{u \in \mathcal{S}} d_M(v, u)$$

候选集 $\mathcal{S}$：等步长下采样，`stride = max(1, V // 64)`，候选数 $\le 64$。
实现：对每个候选点跑一次 Dijkstra，累加到其他候选点的距离，取最小者。

### 区域相容打分 $S_R$

$$S_R(q; P_a) = \lambda_g \tilde r_g(q; c_a) + \lambda_s \tilde r_s(q; c_a) + \lambda_r \tilde r_r(q; c_a)$$

| 项 | 计算方式 |
|---|---|
| $\tilde r_g$ | $d_M(q, c_a) / \sigma_d$ — 测地距离归一化 |
| $\tilde r_s$ | $\|\hat f(q) - \hat f(c_a)\|_2$ — 归一化标量特征差异 |
| $\tilde r_r$ | $1 - n(q) \cdot n(c_a)$ — 法向偏差 |

Chart 通过 Dijkstra BFS，接受 $S_R < \tau_r$ 的所有顶点（不限半径）。

### 前沿打分 $S_{\text{frontier}}$

$$S_{\text{frontier}}(b) = \lambda_U U(b) + \lambda_D D(b) + \lambda_C C(b)$$

- $U(b)$：邻域中未覆盖顶点比例
- $D(b)$：邻域中在 $\Omega$ 之外的顶点比例（外向推进程度）
- $C(b) = \exp(-\text{邻域平均法向变化量})$

### 二点参考邻域 $N_2(b)$

$$S_2(p, q) = w_d \tilde s_d + w_n \tilde s_n + w_f \tilde s_f, \qquad N_2(b) = \{q : S_2(b, q) \le \tau_2\}$$

外向候选集 $E_b = N_2(b^*) \setminus \Omega$。

### 候选质量 $Q(e)$

$$Q(e) = \alpha_U U_P(e) + \alpha_D D_P(e) + \alpha_C C_P(e) - \alpha_F L_{\text{front}}(P(e))$$

| 项 | 说明 |
|---|---|
| $U_P(e) = |P(e) \setminus \Omega| / |P(e)|$ | 新增覆盖度 |
| $C_P(e) = \exp(-\overline{S_R})$ | 区域内部相容稳定度 |
| $L_{\text{front}}(P(e)) = |P(e) \cap \Omega| / |P(e)|$ | 前沿关联代价（惩罚重叠过多） |

### 局部 PCA Frame（确定性）

$$\xi_i = (I - n_a n_a^\top)(x_i - c_a)$$

对 $\{\xi_i\}$ 的 2×2 协方差矩阵用 `np.linalg.eigh`（升序特征值），取**最大特征值**对应方向为 $e_{a,1}$，令 $e_{a,2} = n_a \times e_{a,1}$（右手系）。

四象限划分：$(+,+) \to 0,\ (-,+) \to 1,\ (-,-) \to 2,\ (+,-) \to 3$

### 局部显式结构描述子 $ES_a^{\text{local}}$（53 维）

| 块 | 内容 | 维度 |
|---|---|---|
| **几何块** | 四象限点数占比（4）、空间半径（1）、$H$/$K$ mean/std（4）、冗余（3） | 12 |
| **标量块** | 5 个物理量 × 4 象限 × mean（20）+ 5 个物理量 × 全局 std（5） | 25 |
| **向量块** | 2 个切向梯度场 × 4 象限 × 局部 frame $(e_{a,1}, e_{a,2})$ 投影（2×4×2=16） | 16 |

### 层间法向对应权重（计数法）

$$w_{a \leftarrow b} = \frac{|\{x \in D_a^{k \to k+1} : \Pi_{k \to k+1}(x) \in P_b^{(k+1)}\}|}{|D_a^{k \to k+1}|}$$

实现：对 $P_a^{(k)}$ 中每个顶点沿法向移动 0.5 Bohr，在 $M_{k+1}$ 上用 `cKDTree` 最近邻搜索，统计落入各 chart $P_b^{(k+1)}$ 的比例，归一化得 $\tilde w_{a \leftarrow b}$。

注意方向性：$\tilde w_{a \leftarrow b}^{k \leftarrow k+1} \ne \tilde w_{b \leftarrow a}^{k+1 \leftarrow k}$。

---

## 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tau_r` | 1.0 | 区域相容打分阈值（chart 生长停止条件） |
| `tau_2` | 1.5 | 二点参考邻域阈值 |
| `min_chart_size` | 5 | chart 最小顶点数 |
| `lam` | (0.4, 0.3, 0.3) | $S_R$ 三项权重 $(\lambda_g, \lambda_s, \lambda_r)$ |
| `alpha` | (0.4, 0.3, 0.3, 0.2) | 候选质量 $Q$ 四项权重 $(\alpha_U, \alpha_D, \alpha_C, \alpha_F)$ |

---

## 缓存 I/O

```python
def fclc_cache_path(cache_dir, mol_id, tau_r, tau_2) -> str
    # → {cache_dir}/{mol_id}_fclc_tr{tau_r:.2f}_t2{tau_2:.2f}.pkl

def save_fclc_levels(path, levels: List[FCLCLevel]) -> None
def load_fclc_levels(path) -> List[FCLCLevel]
def load_fclc_entry(path, mol_id) -> List[FCLCLevel]
```

说明：

- 新版 Stage 2 cache 会把 `seed_vertex_idx` 与 `membership_sr` 一起序列化
- 旧版 cache 仍可被读取，但 Stage 3 会在需要时回填这两个字段
- 正式跑 Stage 3/4 时，建议重新生成 Stage 2 cache，避免运行期回填开销

默认合并文件：`{cache_dir}/all_fclc_tr{tau_r:.2f}_t2{tau_2:.2f}.zip`

legacy 合并文件：`{cache_dir}/all_fclc_tr{tau_r:.2f}_t2{tau_2:.2f}.pkl`

---

## 使用示例

```python
from ed2e.data.manifold import load_manifold_levels, manifold_cache_path
from ed2e.data.fclc import build_fclc_levels, fclc_cache_path, save_fclc_levels

# 加载 Stage 1 结果
levels = load_manifold_levels(
    manifold_cache_path("data/ed_energy_5w/cache_manifold", "308", 4, 0.5))

# 构造 FCLC atlas
fclc_levels = build_fclc_levels(
    levels,
    tau_r=1.0,
    tau_2=1.5,
    mem_thresh=3000,   # 可选：控制何时预计算全距离矩阵
)

for lv in fclc_levels:
    print(f"Level {lv.level_id}: {len(lv.charts)} charts")
    for ch in lv.charts[:3]:
        print(f"  Chart {ch.chart_id}: {len(ch.vert_indices)} verts, "
              f"es_local.shape={ch.es_local.shape}")
```

---

## 并行预处理脚本

`scripts/preprocess_stage2.py`：

```bash
# 小批量测试（20 个分子，4 个 worker，默认线程模式）
python scripts/preprocess_stage2.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --cache-dir data/ed_energy_5w/cache_fclc \
    --max-samples 20 --workers 4

# 全量预处理
python scripts/preprocess_stage2.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --cache-dir data/ed_energy_5w/cache_fclc \
    --workers 8

# 显式使用单进程多线程（当前默认）
python scripts/preprocess_stage2.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --cache-dir data/ed_energy_5w/cache_fclc \
    --workers 8 \
    --parallel-mode thread

# 如需切回 fork 进程池
python scripts/preprocess_stage2.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --cache-dir data/ed_energy_5w/cache_fclc \
    --workers 8 \
    --parallel-mode process

# 混合并行：4 个进程，每个进程 2 个 native 计算线程
python scripts/preprocess_stage2.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --cache-dir data/ed_energy_5w/cache_fclc \
    --workers 4 \
    --parallel-mode process \
    --native-threads 2

# 如需 legacy 单个 dict pkl（高内存，不推荐）
python scripts/preprocess_stage2.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --cache-dir data/ed_energy_5w/cache_fclc \
    --workers 4 \
    --merge-format dict

# 内存更保守的长跑配置
python scripts/preprocess_stage2.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --cache-dir data/ed_energy_5w/cache_fclc \
    --workers 8 \
    --parallel-mode process \
    --mem-thresh 2500 \
    --maxtasksperchild 16
```

**并行策略：**
- 默认使用 `ThreadPoolExecutor` + 有界提交，所有线程共享同一份 manifold 数据，并在样本完成后释放对应输入
- 可选使用 `multiprocessing.get_context("fork")` + `Pool.imap_unordered`
- `process` 模式下可通过 `--native-threads` 让每个进程内的 Numba/BLAS/OpenMP 计算使用多个 native 线程，形成“多进程 × 每进程多线程”的混合并行
- Worker 函数：`_worker(mol_id)` → `(mol_id, status, elapsed)`
- `process` 模式下 `imap_unordered` 按完成顺序返回结果，因此进度条中的“第 N 个完成任务”不等于输入顺序中的第 N 个分子
- `process` 模式下支持 `maxtasksperchild` 轮换，降低长时间运行时的内存累积风险
- 默认以流式方式把所有单分子 pkl 打包为 `all_fclc_tr{tau_r:.2f}_t2{tau_2:.2f}.zip`，不再在合并阶段构造全量 FCLC dict
- legacy `dict` 合并格式仍可用，但会在合并时把全量数据读入内存

**命令行参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--manifold-pkl` | 必填 | Stage-1 合并 pkl |
| `--cache-dir` | 必填 | FCLC 缓存目录 |
| `--workers` | `cpu_count/2` | 并行 worker 数（线程或进程） |
| `--parallel-mode` | `thread` | 并行后端；`thread` 为单进程多线程，`process` 为 fork 进程池 |
| `--native-threads` | 1 | `process` 模式下每个 worker 进程可使用的 native 计算线程数，用于混合并行 |
| `--tau-r` | 1.0 | 区域相容阈值 |
| `--tau-2` | 1.5 | 二点邻域阈值 |
| `--min-chart-size` | 5 | 最小 chart 顶点数 |
| `--no-inter` | False | 跳过层间权重计算 |
| `--max-samples` | None | 限制分子数（测试用） |
| `--chunksize` | 2 | `process` 模式下 `imap_unordered` 的分发粒度 |
| `--no-merge` | False | 跳过合并步骤 |
| `--merge-format` | `zip` | merged 输出格式；`zip` 为流式单文件 bundle，`dict` 为 legacy 大 pickle dict |
| `--mem-thresh` | 3000 | 仅对顶点数不超过该阈值的连通分量预计算全距离矩阵；更低值可降低峰值内存 |
| `--maxtasksperchild` | 32 | `process` 模式下每个 worker 处理多少个 chunk 后重建；`0` 表示禁用轮换 |

---

## 消融实验

`scripts/ablate_fclc_chart_size.py` — 扫描 `tau_r` × `tau_2` 网格，统计 chart 大小分布：

```bash
python scripts/ablate_fclc_chart_size.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --n-mols 50 --output-csv data/fclc_ablation.csv
```

**输出 CSV 列：**
`tau_r, tau_2, n_charts_mean, n_charts_std, chart_size_mean, chart_size_median, chart_size_p10, chart_size_p90, coverage_rate, wall_time_s`

**默认扫描网格：**
```
tau_r ∈ [0.5, 0.75, 1.0, 1.25, 1.5]
tau_2 ∈ [1.0, 1.5, 2.0]
```

**目标参数区间：** `chart_size_median ∈ [20, 100]`，`coverage_rate ≈ 1.0`

---

## 可视化

`ed2e/utils/visualize_fclc.py`：

```bash
# plotly 交互 HTML，在浏览器中打开
python ed2e/utils/visualize_fclc.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --fclc-pkl data/ed_energy_5w/cache_fclc/all_fclc_tr1.00_t21.50.zip \
    --mol-id 308 --level 0

# 保存 HTML，叠加原子坐标
python ed2e/utils/visualize_fclc.py ... \
    --atom-pkl data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --save output_fclc.html --no-show

# matplotlib PNG
python ed2e/utils/visualize_fclc.py ... \
    --backend matplotlib --save output_fclc.png
```

**CLI 参数：**

| 参数 | 说明 |
|---|---|
| `--manifold-pkl` | Stage-1 合并 pkl |
| `--fclc-pkl` | Stage-2 合并 bundle/pkl |
| `--mol-id` | 分子 ID |
| `--level` | 密度层编号（默认 0） |
| `--atom-pkl` | 原始 PKL，叠加原子坐标（可选） |
| `--backend` | `plotly`（默认）或 `matplotlib` |
| `--save` | 保存路径 |
| `--no-show` | 不自动弹窗 |

**可视化特性：**
- 每个 chart 用不同颜色渲染（20 色循环调色板）
- 重叠区域（多个 chart 共享顶点的面片）用半透明白边标记
- chart 中心点显示为白色球（plotly 中带 chart_id 标签）
- 支持叠加原子坐标球（按元素着色）

---

## 依赖

```
scipy ≥ 1.7  # sparse.csgraph.shortest_path / dijkstra, spatial.cKDTree
numpy
numba        # JIT 编译前沿计算内核（@njit，首次调用后磁盘缓存）
plotly       # 可选，plotly backend
matplotlib   # 可选，matplotlib backend
tqdm         # 预处理脚本进度条
```

---

## 注意事项

1. **chart 重叠是正常的**：atlas 本质上是局部图册而非互斥分区，重叠区域在 Stage 4 层内传播中用作 overlap-context 修正。
2. **全覆盖保证**：若前沿扩展结束后仍有未覆盖顶点，`build_fclc_atlas` 用最近 chart 中心覆盖它们，确保 `coverage == 1.0`。
3. **vector_features 坐标系**：`es_local` 的向量块在各 chart 自身的局部 frame $(e_{a,1}, e_{a,2})$ 下表示；Stage 3 中使用时无需再做坐标变换。
4. **层间权重方向性**：`inter_weights[a]` 存储的是第 $k$ 层接收端 $a$ 从第 $k+1$ 层发送端接收的归一化权重，不对称。
5. **距离矩阵内存限制**：`shortest_path` 预计算默认仅在 `V ≤ 3000` 时启用；该阈值可通过 `build_fclc_levels(..., mem_thresh=...)`、CLI 的 `--mem-thresh`，或环境变量 `ED2E_FCLC_MEM_THRESH` 调整。阈值越高，速度通常更快，但峰值内存也更高。
6. **线程模式的适用性**：`thread` 模式不会复制 manifold 到多个子进程，且现在会在样本完成后逐步释放对应输入，通常更适合内存紧张或 `fork` 模式不稳定的场景。是否能明显提速取决于 NumPy/SciPy/Numba 计算释放 GIL 的比例。
7. **默认 merged 输出已改为流式 bundle**：`zip` 格式不会在合并阶段构造一个全量 FCLC dict，因此单文件输出不再等价于“全量内存合并”。如果下游只需要某个 `mol_id`，应优先使用 `load_fclc_entry(...)`。
8. **混合并行的推荐起点**：如果想用“多个进程、每个进程多个线程”，建议先从 `--parallel-mode process --workers 4 --native-threads 2` 或 `--workers 3 --native-threads 2` 起步，避免总线程数过高导致反而变慢。
9. **长跑稳定性**：对于大数据集，如果使用 `process` 模式，推荐保留 `--maxtasksperchild` 默认值或进一步减小它，以便周期性重建 worker，避免处理大量大分子后子进程内存持续膨胀。
10. **Numba 首次编译延迟**：第一个分子处理时 `@njit` 函数会触发编译（数秒），之后从磁盘缓存加载，不影响批量吞吐量。
