# Stage 3: FCLC 内聚合与 Stage 4 层内聚合接口

## 当前范围

本阶段实现的是 **Stage 3 local block**，但所有数据组织、缓存格式、运行态输出都按“**下一步立刻进入 Stage 4 intra block**”设计。

当前已经落地的内容：

- Stage 2 -> Stage 3 所需接口升级
- Stage 3 单分子 flat sample/cache 构建
- Stage 3 面向训练的 packed mmap cache
- Stage 3 local block
- Stage 4 所需静态 bundle 的预构建与零重排消费接口
- 无 readout 的 smoke forward
- 数据流检查函数与 CLI

当前明确 **不做**：

- 全局读出
- 能量头
- loss
- 真正的 Stage 4 主传播
- 后续层间传播

`chart self-update` 本阶段没有作为单独模块存在。原因是这里采用的是：

1. 由当前 local state 编码出 `(\tilde F_a^{local}, p_a)`
2. 用 `\tilde F_a^{local}` 调制 FCLC 内消息传递
3. 用更新后的 local state 重新编码出下一个 `(\tilde F_a^{local}, p_a)`

也就是说，`T_chart_encode` 已经承担了“下一轮 local / 下一阶段 intra 的 chart state 起点”这一职责，因此不再额外插入一个独立 `chart self-update`。

---

## 为什么 Stage 3 / 4 一体化设计

Stage 3 和 Stage 4 不是两个松散模块，而是同一条前向链路上的连续两段。

本实现遵循以下约束：

- Stage 3 运行结束后，直接产出 Stage 4 可消费状态，而不是先构造高层 Python 对象
- Stage 4 所需静态结构在预处理阶段一次性物化
- 训练/前向时不重新构图
- 所有索引预先压平成整数张量空间
- local 输出张量布局直接匹配 intra 输入

因此，local block 输出固定为：

- `node_state_shared_next`
- `local_state_final`
- `p_next_local`
- `intra_static_bundle`

其中 `intra_static_bundle` 直接来自 batch，不发生重排。

---

## 三层状态与坐标系

当前实现固定三层状态：

1. 物理节点共享状态 `h_i^(t)`
2. chart 内局部节点状态 `\tilde h_{a,i}^(t)`
3. chart 状态 `p_a^(t)`

三层状态全部采用 **标量 + 向量双流**。

### 1. 物理节点共享状态

- 标量部分：`(N, C_s)`
- 向量部分：`(N, C_v, 2)`
- 向量基：**节点自己的切平面基**

### 2. chart 内局部节点状态

- 标量部分：`(M, C_s)`
- 向量部分：`(M, C_v, 2)`
- 向量基：**chart 中心的统一 local frame**

也就是说，对同一个 chart `P_a`，所有局部节点向量都统一表示在

- `chart_frame = (e_{a,1}, e_{a,2})`

下，而不是再为每个 membership 单独构造一套局部 basis。

这里采用的是一致的 flat-chart 假设：

- `local_coords` 已经把 chart 近似成二维平面
- `local_state.vector` 也沿用同一个二维 chart frame
- 从曲面到 FCLC 平面的误差，在当前阶段统一视作局部线性近似误差

### 3. chart 状态

- 标量部分：`(A, C_s)`
- 向量部分：`(A, C_v, 2)`

这里 `A` 是 chart 数。

chart 状态同样是双流，因为从显式结构编码得到的 `\tilde F` 和 `p` 都需要同时保留标量调制与向量调制信息。

---

## 节点输入性质与保守 v1

主网络的底层节点不是原子，也不是体素，而是：

- 多层等值流形上的离散采样点

当前保守实现保留的节点输入：

### 几何基础

- `x_i`
- `n_i`

### 标量通道

- `||∇ρ||`
- `Δρ`
- `H`
- `K`
- `∂_n^2 ρ`

### 向量通道

- `∇_M ||∇ρ||`
- `∇_M H`

因此节点原始输入写成：

- `h_i^(0) = (h_i^{s,(0)}, h_i^{v,(0)})`

其中：

- `h_i^{s,(0)} = (||∇ρ||, Δρ, H, K, ∂_n^2 ρ)`
- `h_i^{v,(0)} = (∇_M ||∇ρ||, ∇_M H)`

当前版刻意不引入更重的电子结构量、额外张量场和全局条件头，目的是先验证：

- 多层等值流形
- FCLC atlas
- local -> intra 接口

这条主线。

---

## Stage 2 接口升级

为支持 Stage 3，`FCLCChart` 新增了两项字段：

- `seed_vertex_idx`
- `membership_sr`

含义如下：

- `seed_vertex_idx`：该 chart 的原始种子顶点，局部索引，相对于所属 `ManifoldComponent`
- `membership_sr`：该 chart 每个成员点相对于种子点的 `S_R` 分数，顺序与 `vert_indices` 对齐

如果读取的是旧版 Stage 2 cache：

- `seed_vertex_idx` 缺失时，Stage 3 用“距 chart center 最近的 component 顶点”回填
- `membership_sr` 缺失时，Stage 3 基于 Stage 1 component 的几何和标量特征重新计算

这保证 Stage 3 对旧 cache 有有限兼容，但正式运行仍建议重新生成 Stage 2 cache。

---

## Stage 3 Sample / Cache Schema

单分子 sample 在 [`ed2e/data/stage3_local.py`](/Users/sii-haoyutang/Code/PycharmProjects/ED2E/ed2e/data/stage3_local.py) 中定义为 `Stage3Sample`。

### 节点级字段

- `node_xyz`: `(N, 3)`
- `node_normal`: `(N, 3)`
- `node_scalar_raw`: `(N, 5)`
- `node_vector_raw`: `(N, 2, 3)`
- `node_tangent_basis`: `(N, 2, 3)`

### membership 级字段

- `chart_membership`: `(M, 2)`
  - 第 0 列：chart 索引
  - 第 1 列：物理节点索引
- `membership_sr`: `(M,)`
- `membership_weight`: `(M,)`
  - 由 `exp(-membership_sr)` 在同一物理节点上归一化得到
- `local_coords`: `(M, 2)`
- `quadrant`: `(M,)`

### local 图字段

- `local_knn_edge_index`: `(2, E_local)`
- `local_edge_attr`: `(E_local, 6)`

`local_edge_attr` 当前为：

- `dx`
- `dy`
- `dist`
- `same_quadrant`
- `q_src / 3`
- `q_dst / 3`

### chart 级显式结构 / anchor 字段

- `chart_es_geom_static`: `(A, 32)`
- `chart_anchor_pos`: `(A, 8, 2)`
- `chart_anchor_mask`: `(A, 8)`

当前 `chart_es_geom_static` 由以下部分拼接而成：

- 四块占据比例
- 四块几何质心
- 四块协方差不变量 `trace / det`
- 四块 `H` 均值
- 四块 `K` 均值
- 四块法向分散度

### Stage 4 静态图字段

- `reference_chart_id`: `(G,)`
- `chart_graph_edge_index`: `(2, E_chart)`
- `overlap_edge_index`: `(2, E_overlap)`
- `overlap_shared_membership_index`: `(S_overlap, 2)`
- `overlap_shared_ptr`: `(E_overlap + 1,)`
- `overlap_jaccard`: `(E_overlap,)`

这里：

- `G` 是 chart-group 数，不是全局 level 数
- 当前实现把 **同一 level 且同一 manifold component** 视为一个 chart-group
- `reference_chart_id` 对每个 chart-group 存一个 geodesic medoid

之所以这样做，是因为跨不同 connected component 的 geodesic medoid 本身并不良定义。

### 元数据字段

`chart_frame_metadata` 是一个 dict，目前包含：

- `chart_center`
- `chart_center_normal`
- `chart_frame`
- `chart_level_id`
- `chart_component_id`
- `chart_group_id`
- `chart_seed_node_index`
- `chart_stage2_id`
- `group_level_id`
- `group_component_id`

---

## 训练优先的 Packed Cache

单分子 `pkl` 和 merged `zip bundle` 适合调试，不适合作为正式训练的主读取路径。

主要原因是：

- 每次取样都要做一次 Python `pickle` 反序列化
- `zip bundle` 还会额外引入 archive member 查找
- DataLoader 多 worker 时，这条路径的 Python 开销会被反复放大

因此当前已经新增 **packed mmap cache**，定义在 [`ed2e/data/stage3_packed.py`](/Users/sii-haoyutang/Code/PycharmProjects/ED2E/ed2e/data/stage3_packed.py)。

其设计原则是：

- 每个字段单独落成一个扁平 `.npy`
- 每个分子的边界通过 `index.npz` 中的 pointer arrays 描述
- worker 首次访问时懒打开 memmap，后续只做 slice
- 训练热路径不再做逐样本 pickle 反序列化

### Packed 目录结构

packed 输出目录当前固定包含：

- `meta.json`
- `index.npz`
- `node_xyz.npy`
- `node_normal.npy`
- `node_scalar_raw.npy`
- `node_vector_raw.npy`
- `node_tangent_basis.npy`
- `chart_membership.npy`
- `membership_sr.npy`
- `membership_weight.npy`
- `local_coords.npy`
- `quadrant.npy`
- `local_knn_edge_index.npy`
- `local_edge_attr.npy`
- `chart_es_geom_static.npy`
- `chart_anchor_pos.npy`
- `chart_anchor_mask.npy`
- `reference_chart_id.npy`
- `chart_graph_edge_index.npy`
- `overlap_edge_index.npy`
- `overlap_shared_membership_index.npy`
- `overlap_shared_ptr.npy`
- `overlap_jaccard.npy`
- `chart_center.npy`
- `chart_center_normal.npy`
- `chart_frame.npy`
- `chart_level_id.npy`
- `chart_component_id.npy`
- `chart_group_id.npy`
- `chart_seed_node_index.npy`
- `chart_stage2_id.npy`
- `group_level_id.npy`
- `group_component_id.npy`

其中：

- `meta.json` 记录 packed format、字段文件名、dtype、尾部 shape
- `index.npz` 记录 `mol_ids` 与各实体维度的 pointer arrays
- 所有大数组都可以通过 `np.load(..., mmap_mode="r")` 只读打开

### 为什么它更适合训练

- 读取路径从“反序列化复杂 Python 对象”变成“切一段连续数组”
- OS page cache 更容易复用这些顺序数组访问
- 局部索引字段在 disk 上收紧为 `int32`，减少无意义 IO
- `Stage3PackedDataset` 在 worker 内只保留 index 常驻内存，大数组始终走 mmap

当前推荐：

- 调试：继续用单分子 `pkl` 或 `zip bundle`
- 训练：优先使用 packed mmap cache

---

## 函数职责与接口

### 数据层：`ed2e/data/stage3_local.py`

#### `build_stage3_sample(...) -> Stage3Sample`

职责：

- 将 Stage 1 manifold levels 与 Stage 2 fclc levels 组装成单分子 flat sample
- 构造 membership 权重
- 构造 local kNN 图
- 构造 chart 图与 overlap 边
- 构造 reference chart / anchors / metadata

关键参数：

- `local_knn_k=12`
- `chart_knn_k=8`
- `num_anchors=8`
- `inner_threads`

#### `collate_stage3_samples(...) -> Stage3TensorBatch`

职责：

- 多分子 batch 拼接
- 统一节点、chart、membership、group 的偏移量
- 直接产出可上设备张量

训练相关补充：

- `Stage3TensorBatch` 现在支持 `pin_memory()`
- `Stage3TensorBatch` 现在支持 `.to(device, non_blocking=True/False)`

因此后续训练脚本可以直接使用自定义 batch，而不需要手工递归搬运每个字段。

### 数据层：`ed2e/data/stage3_packed.py`

#### `pack_stage3_cache(...)`

职责：

- 从 Stage 3 `pkl` 目录、单个 `zip bundle` 或单分子 `pkl`
- 生成训练用 packed mmap cache

流程：

- 第一遍扫描统计所有实体维度的总长度，并检查 schema 一致性
- 第二遍按字段流式写入扁平 `.npy`
- 输出 `meta.json + index.npz`

#### `Stage3PackedDataset(...)`

职责：

- 惰性打开 packed `.npy`
- 在 `__getitem__` 中按 pointer slice 还原单分子 `Stage3Sample`

关键行为：

- index 常驻内存
- 大数组全部只读 mmap
- 在 worker fork/spawn 时，不直接携带已打开的 mmap handle

#### `load_stage3_packed_sample(...)`

职责：

- 通过 `mol_id` 读取 packed cache 中的单分子 sample

适用场景：

- smoke
- debug
- pkl / packed 一致性比对

#### `validate_stage3_sample(...)`

职责：

- 汇总 sample shape 与数据流检查结果

#### `check_membership_weights(...)`

检查：

- 同一物理节点上的 membership weight 是否归一为 1

#### `check_overlap_jaccard(...)`

检查：

- overlap 共享点集与 `overlap_jaccard` 是否一致

#### `check_chart_plane_residual(...)`

检查：

- 节点到 chart 中心平面的离平面残差

#### `check_chart_normal_alignment(...)`

检查：

- 节点法向与 chart 中心法向的偏离

#### `check_chart_vector_projection(...)`

检查：

- 将节点原始 3D 向量投到 chart 统一 frame 后的近似损失

### 模型层：`ed2e/model/stage3_local.py`

#### `FCLCLocalBlock`

职责：

- 初始化共享节点双流状态
- 初始化 local membership 双流状态
- 从 local state 动态刷新 `ES_scalar / ES_vector`
- 做 `ES -> \tilde F / p`
- 做 local 消息传递
- merge 回共享节点状态
- 输出 Stage 4-ready 接口

输出字段：

- `node_state_shared_next`
- `local_state_final`
- `p_next_local`
- `intra_static_bundle`

#### `ExplicitStructureEncoder`

职责：

- `Enc_g / Enc_s / Enc_v`
- 三类 token 的 block-token attention
- 输出双流 `mod_state` 与双流 `chart_state`

#### `PseudoStage4Consumer`

职责：

- 不实现真正 Stage 4 主传播
- 只验证 `p_next_local + intra_static_bundle + local_state_final + node_state_shared_next` 是否可以零重排消费

### 脚本层

#### `scripts/preprocess_stage3.py`

职责：

- 离线构建 Stage 3 cache
- 外层进程 / 线程并行
- 可选 merge 为 zip bundle
- 可选同步生成 packed mmap cache

新增参数：

- `--packed-dir`
- `--packed-overwrite`

推荐把 packed 输出视为正式训练入口，而不是事后再从 `pkl` 直接训练。

#### `scripts/pack_stage3_cache.py`

职责：

- 将已有 Stage 3 cache 重新打包成训练用 packed cache

适合：

- 已有 `cache_stage3/` 目录
- 只有 merged `zip bundle`
- 需要从旧缓存补做训练用 mmap store

#### `scripts/smoke_stage3_local_forward.py`

职责：

- 构建或加载单分子 Stage 3 sample
- 跑 local block
- 跑伪 Stage 4 consumer
- 输出 shape / 动态 ES 刷新 / zero-repack 检查
- 现在额外支持从 `--stage3-packed-dir` 直接读取

---

## local block 的 5 个步骤

当前 `FCLCLocalBlock` 的前向分为五步。

### Step 1. `T_init`

从共享节点状态初始化 chart 内局部状态：

- 标量流直接 gather
- 向量流从 `node_tangent_basis` 变换到当前 chart 的统一 `chart_frame`

### Step 2. 动态 ES 刷新

每一轮 outer block 根据当前 local state 重新统计：

- `ES_geom`：静态，不变
- `ES_scalar`：随 local scalar state 刷新
- `ES_vector`：随 local vector state 刷新

### Step 3. `ES -> \tilde F / p`

流程固定为：

- `Enc_g`
- `Enc_s`
- `Enc_v`
- 三类 token 做一次 typed token attention
- 输出双流 `mod_state`
- 输出双流 `chart_state`

这里没有把 ES 本体送入网络之外的额外黑箱；只有 `ES -> \tilde F / p` 这一步经过网络。

### Step 4. FCLC 内 local message passing

当前实现采用：

- chart 级共享 gate
- chart 级 attention bias
- scalar / vector 双流消息

其中：

- attention 由边特征与节点特征共同决定
- chart 级 `mod_state` 对消息核做共享调制

### Step 5. `T_merge`

把 local state merge 回共享节点状态：

- 标量流按 `membership_weight` 聚合
- 向量流先从 chart 的统一 `chart_frame` 变回 `node_tangent_basis`，再按 `membership_weight` 聚合

---

## 显式结构描述子与 `ES -> \tilde F`

### `ES_geom`

当前已实现为静态块：

- 四块占据比例
- 四块几何质心
- 四块协方差不变量
- 四块 `H/K` 均值
- 四块法向分散度

### `ES_scalar`

从当前 local scalar state 统计：

- 四块均值
- 四块方差
- 对置块差 `(0,2)` 与 `(1,3)`

### `ES_vector`

从当前 local vector state 统计：

- 四块向量均值
- 四块模长均值
- 四块离散度
- 对置块向量差
- 对置块向量内积

### `ES -> \tilde F / p`

当前实现是：

1. `Enc_g(ES_geom)`
2. `Enc_s(ES_scalar)`
3. `Enc_v(ES_vector)`
4. 三个 token 做一次 `MultiheadAttention`
5. 输出：
   - `mod_scalar`
   - `mod_vector`
   - `chart_scalar`
   - `chart_vector`

因此当前已经满足：

- `F` 有标量部分和向量部分
- chart state `p` 也有标量部分和向量部分
- 标量/向量与 attention 的交互都发生在 `ES -> \tilde F / p` 阶段

---

## Stage 4 接口 bundle

`intra_static_bundle` 当前固定包含：

- `chart_graph_edge_index`
- `overlap_edge_index`
- `overlap_shared_membership_index`
- `overlap_shared_ptr`
- `overlap_jaccard`
- `reference_chart_id`
- `chart_anchor_pos`
- `chart_anchor_mask`
- `chart_frame_metadata`

### 低延迟设计理由

- Stage 4 不再重建 chart 图
- overlap 不再现场查共享点
- reference chart 已预计算
- anchor 已预计算
- overlap-context 所需共享 membership 对已预展开
- batch 内所有对象都已经是连续整数索引

因此从 Stage 3 到 Stage 4 的接口切换只需要张量引用，不需要对象解包。

---

## 当前对 Stage 4 的静态定义

### chart 图

- `chart_graph_edge_index = geodesic kNN ∪ overlap edges`
- `geodesic chart kNN: k = 8`

### overlap

- overlap 边是 **有向** 的
- `a <- b` 与 `b <- a` 分别存储
- overlap 标量为 `Jaccard`

### overlap-context token

对于 `a <- b`，当前实现只使用“接收端视角”：

- 共享点的共享标量
- 共享点在当前 FCLC `a` 中的 local 向量
- 共享点在当前 FCLC `a` 中的 local 坐标

这与前面的设计讨论保持一致，不在当前阶段把两侧向量同时塞进一个 token。

### reference chart

实现上采用 **chart-group 内的 geodesic medoid**，chart-group 定义为：

- 同一 `level_id`
- 同一 `component_id`

这是对“reference chart: geodesic medoid”的一个必要收紧，因为跨 disconnected component 的 geodesic medoid 不可直接定义。

---

## 并行化与内存策略

`scripts/preprocess_stage3.py` 当前采用分层混合并行：

- 外层：少量 worker，对分子并行
- 内层：单分子内部线程池，对 chart local 静态结构构建并行

默认参数：

- `local_knn_k = 12`
- `chart_knn_k = 8`
- `num_anchors = 8`

内存策略：

- 正式路径优先 `--manifold-cache-dir`，逐分子读取 Stage 1 cache
- Stage 2 支持 cache dir 或 bundle 按 mol_id 读取
- 训练阶段优先 packed mmap cache，而不是重新打开 `pkl` / `zip`
- 训练/前向阶段不重构 local 图与 chart 图
- smoke/debug 额外支持 `--manifold-pkl`，仅用于当前仓库这类只保留 merged Stage 1 文件的情况

---

## CLI

### 1. 离线构建 Stage 3 cache

推荐路径，逐分子读取 Stage 1 cache：

```bash
python scripts/preprocess_stage3.py \
    --manifold-cache-dir data/ed_energy_5w/cache_manifold \
    --fclc-source data/ed_energy_5w/cache_fclc \
    --cache-dir data/ed_energy_5w/cache_stage3 \
    --packed-dir data/ed_energy_5w/cache_stage3_packed \
    --workers 4 \
    --threads-per-proc 4
```

如果当前只有 merged Stage 1 文件，可用回退模式：

```bash
python scripts/preprocess_stage3.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --fclc-source data/ed_energy_5w/cache_fclc \
    --cache-dir data/ed_energy_5w/cache_stage3 \
    --packed-dir data/ed_energy_5w/cache_stage3_packed \
    --workers 2 \
    --threads-per-proc 2
```

如果已有旧版 Stage 3 cache，也可以单独补打包：

```bash
python scripts/pack_stage3_cache.py \
    --stage3-source data/ed_energy_5w/cache_stage3 \
    --packed-dir data/ed_energy_5w/cache_stage3_packed
```

### 2. local -> pseudo intra smoke

从已存在的 Stage 3 cache 读取：

```bash
python scripts/smoke_stage3_local_forward.py \
    --mol-id 482085 \
    --stage3-source data/ed_energy_5w/cache_stage3
```

从 packed cache 读取：

```bash
python scripts/smoke_stage3_local_forward.py \
    --mol-id 482085 \
    --stage3-packed-dir data/ed_energy_5w/cache_stage3_packed
```

从 Stage 1 + Stage 2 现构单分子 sample：

```bash
python scripts/smoke_stage3_local_forward.py \
    --mol-id 482085 \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --fclc-source data/ed_energy_5w/cache_fclc \
    --steps 2 \
    --verify-single-and-multi
```

---

## 检查数据流的方法

本阶段要求至少检查以下八项。

### 1. 单分子 sample 字段与 shape

方法：

- 调用 `summarize_stage3_sample(sample)`
- 核对：
  - `num_nodes`
  - `num_charts`
  - `num_memberships`
  - `num_local_edges`
  - `num_chart_edges`
  - `num_overlap_edges`
  - `num_overlap_pairs`

### 2. `membership_sr / membership_weight`

方法：

- 调用 `check_membership_weights(sample)`
- 期待：
  - 同一物理节点上的 `membership_weight` 求和为 1
  - `max_abs_error` 足够小

当前真实 smoke 示例中，误差量级约为 `1e-7`。

### 3. overlap 共享索引与 Jaccard

方法：

- 调用 `check_overlap_jaccard(sample)`
- 重新从 `chart_membership` 求交并验证：
  - `overlap_shared_membership_index`
  - `overlap_jaccard`

### 4. flat-chart 近似质量检查

方法：

- 调用：
  - `check_chart_plane_residual(sample)`
  - `check_chart_normal_alignment(sample)`
  - `check_chart_vector_projection(sample)`

这里不再把“节点基 -> chart 基 -> 节点基”的精确往返当作目标，而是直接检查当前 flat-chart 假设是否足够合理：

- 点是否足够接近 chart 平面
- 节点法向是否足够接近 chart 中心法向
- 节点向量投到统一 chart frame 后损失是否可接受

### 5. 动态 ES 刷新

方法：

- 运行 `smoke_stage3_local_forward.py`
- 观察：
  - `geom_static_stable=True`
  - `scalar_es_refreshed=True`
  - `vector_es_refreshed=True`

这表示：

- 几何块不随 local propagation 变化
- 标量块与向量块确实会按当前 local state 动态刷新

### 6. 单轮 / 多轮 local smoke

方法：

- `--steps 2`
- 再加 `--verify-single-and-multi`

观察：

- `step1_chart_scalar` 与 `stepN_chart_scalar` shape 一致
- local edges 数与 batch shape 自洽
- 单轮与多轮都能完成前向

### 7. Stage 3 输出到伪 Stage 4 输入的零重排检查

方法：

- 运行 smoke 脚本
- 观察：
  - `zero_repack=True`
  - `pseudo_stage4_identity=True`

含义：

- Stage 4 消费器直接使用 Stage 3 输出的 `intra_static_bundle`
- 没有额外重排，也没有中间 Python 对象解包

### 8. Packed cache 一致性与训练入口检查

方法：

- 用同一个 `mol_id` 分别从 `pkl` 与 packed cache 读取
- 逐字段比较：
  - `node_*`
  - `membership_*`
  - `local_*`
  - `chart_*`
  - `overlap_*`
  - `chart_frame_metadata`

注意：

- packed 内部局部索引字段可以收紧成 `int32`
- 只要数值与 shape 一致，就视为通过

训练前还建议至少确认一次：

- `Stage3PackedDataset(...)`
- `DataLoader(..., collate_fn=collate_stage3_samples, pin_memory=True)`
- batch 类型应为 `Stage3TensorBatch`
- batch 应支持：
  - `batch.pin_memory()`
  - `batch.to(device, non_blocking=True)`

---

## 已实现但属于下一阶段的接口约束

虽然本轮不实现真正的 Stage 4 主传播，但已经把与下一阶段有关的接口固定下来：

- chart 图边集
- overlap 边与共享 membership 对
- reference chart
- anchor 编码
- chart 双流状态 `p_next_local`
- zero-repack `intra_static_bundle`
- packed 训练 cache 的字段与 slice 边界

也就是说，下一阶段的主要工作不再是“重新谈接口”，而是：

- 在当前 `p_next_local + intra_static_bundle` 之上实现真正的 intra-chart message passing
- 再继续接层间传播

---

## 当前真实 smoke 结果

在当前仓库真实数据上，对 `mol_id=482085` 运行：

```bash
python scripts/smoke_stage3_local_forward.py \
    --mol-id 482085 \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --fclc-source data/ed_energy_5w/cache_fclc \
    --steps 2 \
    --verify-single-and-multi
```

已确认：

- `membership_weight` 检查通过
- `overlap_jaccard` 检查通过
- `chart_plane_residual` 检查通过
- `chart_normal_alignment` 检查通过
- `chart_vector_projection` 检查通过
- `geom_static_stable=True`
- `scalar_es_refreshed=True`
- `vector_es_refreshed=True`
- `zero_repack=True`
- `pseudo_stage4_identity=True`

这说明当前实现已经满足：

- Stage 3 sample/cache 构建成立
- Stage 3 local block 前向成立
- Stage 3 -> Stage 4 的静态接口可直接消费
