# Stage 1: Multi-Level Isodensity Manifold Extraction

## 概述

本阶段将每个分子的电子密度点云 `(coords, densities)` 转换为多层等值流形面表示。

**核心方法：** 先从 PKL 点云重建原始 3D 体素网格，再用 Marching Cubes 算法提取三角化等值面，最后用三次 B-spline 插值计算法向量、曲率和密度梯度等物理量。

**输入**（来自 `EDBenchPKLDataset._raw`）：
- `coords`：`(M, 3)` float32，密度点坐标（Bohr）
- `densities`：`(M,)` float32，密度值（e/Bohr³），已过滤至 > 0.05

**输出**：
- `List[ManifoldLevel]`：K 层等值流形，level_id 0 为最内层（高密度），level_id K-1 为最外层（低密度）

---

## 文件位置

```
ed2e/
└── data/
    └── manifold.py    ← 本阶段全部代码
```

---

## 完整流水线

```
(coords, densities)
    │
    ├─ reconstruct_density_grid()       → DensityGrid
    │
    ├─ smooth_density_grid()            → DensityGrid（高斯预平滑）
    │
    ├─ select_thresholds_percentile()   → [c_0, ..., c_{K-1}]（降序）
    │
    └─ 对每个阈值 c_k：
        ├─ extract_isosurface_mesh()                  → verts, faces, normals
        ├─ compute_density_derivatives_bspline()       → grad_rho, laplacian, d2n_rho
        ├─ compute_mesh_curvatures()                   → H, K
        ├─ assemble_point_features()                   → scalar_features, vector_features
        └─ find_mesh_components()                      → List[ManifoldComponent]
                                                         → ManifoldLevel
```

---

## 数据结构

### `DensityGrid`

重建的 3D 密度体素网格。

```python
@dataclass
class DensityGrid:
    density: np.ndarray  # (Nx, Ny, Nz) float32 — 密度值；缺失体素填 0
    origin:  np.ndarray  # (3,) float32 — 网格原点（Bohr）
    spacing: np.ndarray  # (3,) float32 — 各轴网格间距（Bohr）
```

---

### `ManifoldComponent`

一个等值流形连通分量 $M_{k,\ell}$，保留完整三角网格。

```python
@dataclass
class ManifoldComponent:
    verts:           np.ndarray  # (V, 3)    float32 — 顶点坐标（Bohr）
    faces:           np.ndarray  # (F, 3)    int32   — 三角面片（局部顶点索引）
    normals:         np.ndarray  # (V, 3)    float32 — 单位法向量，指向 ∇ρ > 0
    scalar_features: np.ndarray  # (V, 5)    float32 — 见下表
    vector_features: np.ndarray  # (V, 2, 3) float32 — 见下表
    component_id:    int
    density_level:   float       — 所属密度阈值 c_k
```

**`scalar_features` 通道（对应论文 §A.1.2 中的 u_s）：**

| 索引 | 符号 | 含义 | 计算方式 |
|------|------|------|----------|
| 0 | $\|\nabla\rho\|$ | 密度梯度模长 | B-spline 有限差分 |
| 1 | $\Delta\rho$ | 密度 Laplacian | B-spline 有限差分 |
| 2 | $H$ | 平均曲率 | cotangent Laplacian |
| 3 | $K$ | Gaussian 曲率 | 离散角亏量公式 |
| 4 | $\partial_n^2\rho$ | 法向二阶导数 | B-spline 沿法向差分 |

**`vector_features` 通道（对应论文 §A.1.2 中的 u_v）：**

| 索引 | 符号 | 含义 |
|------|------|------|
| 0 | $\nabla_{M_k}\|\nabla\rho\|$ | $\|\nabla\rho\|$ 在流形切平面上的梯度 |
| 1 | $\nabla_{M_k} H$ | 平均曲率在流形切平面上的梯度 |

---

### `ManifoldLevel`

一个密度层 $M_k = \bigsqcup_\ell M_{k,\ell}$。

```python
@dataclass
class ManifoldLevel:
    level_id:   int                      — 层编号（0 = 最内层）
    threshold:  float                    — 密度阈值 c_k
    components: List[ManifoldComponent]  — 该层所有连通分量
```

---

## 函数接口

### 主入口

```python
def extract_manifold_levels(
    coords:             np.ndarray,
    densities:          np.ndarray,
    n_levels:           int = 4,
    percentiles:        Optional[List[float]] = None,  # 默认 [20,40,60,80]
    smooth_sigma:       float = 0.5,   # Bohr；0 = 不平滑
    min_component_size: int = 10,
) -> List[ManifoldLevel]:
```

---

### 步骤函数

#### `reconstruct_density_grid`

```python
def reconstruct_density_grid(
    coords: np.ndarray,    # (M, 3) float32
    densities: np.ndarray, # (M,) float32
) -> DensityGrid:
```

从过滤点云反推原始体素网格参数（间距、原点、维度），将已知密度值填入 3D 数组，其余位置填 0。

**原理：** PKL 点云来自 Gaussian cube 文件的规则栅格，坐标 = origin + i·Δ，因此各轴坐标的最小相邻差值即为网格间距。

---

#### `smooth_density_grid`

```python
def smooth_density_grid(
    grid: DensityGrid,
    sigma_bohr: float = 0.5,
) -> DensityGrid:
```

对体素网格做各向同性高斯平滑（`scipy.ndimage.gaussian_filter`），消除填 0 空体素造成的跳跃伪影，使等值面更光滑。

---

#### `select_thresholds_percentile`

```python
def select_thresholds_percentile(
    densities: np.ndarray,
    n_levels: int = 4,
    percentiles: Optional[List[float]] = None,
) -> np.ndarray:  # (n_levels,) 降序
```

按密度百分位选取 K 个阈值，返回降序数组（level_id 0 对应最高密度）。

---

#### `extract_isosurface_mesh`

```python
def extract_isosurface_mesh(
    grid: DensityGrid,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 返回 (verts, faces, normals)
    # verts:   (V, 3) float32 — Bohr 绝对坐标
    # faces:   (F, 3) int32
    # normals: (V, 3) float32 — 指向 ∇ρ > 0
```

调用 `skimage.measure.marching_cubes`，提取三角化等值面。对法向量做方向校验：采样若干顶点处的有限差分梯度，若 normals · ∇ρ < 0 则全部取反。

---

#### `compute_density_derivatives_bspline`

```python
def compute_density_derivatives_bspline(
    grid:           DensityGrid,
    verts:          np.ndarray,         # (V, 3) Bohr
    normals:        np.ndarray,         # (V, 3)
    h_normal:       float = 0.1,        # Bohr
    bspline_coeffs: Optional[np.ndarray] = None,  # 跨层复用
) -> Dict[str, np.ndarray]:
    # keys: "grad_rho" (V,3), "grad_norm" (V,), "laplacian" (V,), "d2n_rho" (V,)
```

用 `scipy.ndimage.spline_filter` 预计算三次 B-spline 系数，再用 `map_coordinates` 在网格坐标系内做中心差分，精确估计 ∇ρ 和 Δρ。同一分子的 `bspline_coeffs` 只需计算一次，可传入所有层复用。

---

#### `compute_mesh_curvatures`

```python
def compute_mesh_curvatures(
    verts:   np.ndarray,  # (V, 3)
    faces:   np.ndarray,  # (F, 3)
    normals: np.ndarray,  # (V, 3)
) -> Tuple[np.ndarray, np.ndarray]:  # (H, K)，均为 (V,) float32
```

**平均曲率 H：** 使用余切 Laplacian（Desbrun et al. 1999），对每条边累积余切权重，除以混合面积，取符号后得有符号 H。

**Gaussian 曲率 K：** 角亏量公式 $K(i) = (2\pi - \sum_T \theta_i^T) / A_i$。

计算全向量化，无 Python 面循环。

---

#### `compute_tangential_gradient`

```python
def compute_tangential_gradient(
    field_vals: np.ndarray,  # (V,)
    verts:      np.ndarray,  # (V, 3)
    faces:      np.ndarray,  # (F, 3)
    normals:    np.ndarray,  # (V, 3)
) -> np.ndarray:             # (V, 3) float32 — ∇_M f
```

对每个三角面求解 2×2 线性系统 `∇f · e₁ = f_j − f_i, ∇f · e₂ = f_k − f_i`，按面积加权平均到顶点，投影到切平面。

---

#### `assemble_point_features`

```python
def assemble_point_features(
    verts, faces, normals, deriv_dict, H, K
) -> Tuple[np.ndarray, np.ndarray]:
    # (scalar_features (V,5), vector_features (V,2,3))
```

将各步骤结果打包为 `scalar_features` 和 `vector_features`。

---

#### `find_mesh_components`

```python
def find_mesh_components(
    verts:    np.ndarray,  # (V, 3)
    faces:    np.ndarray,  # (F, 3)
    min_size: int = 10,
) -> List[Dict]:
    # 每个 dict: {"vert_idx": (Vc,), "local_faces": (Fc, 3)}
```

用 `scipy.sparse.csgraph.connected_components` 在顶点邻接图上分解，过滤小于 `min_size` 的噪声分量，返回局部重索引的 face 数组。

---

## 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_levels` | 4 | 层数 K |
| `percentiles` | `[20,40,60,80]` | 密度百分位阈值 |
| `smooth_sigma` | 0.5 Bohr | Gaussian 预平滑宽度 |
| `min_component_size` | 10 | 最小连通分量顶点数 |
| `h_normal` | 0.1 Bohr | 法向二阶导有限差分步长 |

---

## 依赖

```
scikit-image  ≥ 0.19   # skimage.measure.marching_cubes
scipy         ≥ 1.7    # ndimage.spline_filter / map_coordinates / sparse.csgraph
numpy
tqdm                   # 预处理脚本进度条
```

---

## 使用示例

```python
from ed2e.data import EDBenchPKLDataset
from ed2e.data.manifold import extract_manifold_levels

dataset = EDBenchPKLDataset(
    pkl_path="data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl",
    cache_dir="data/ed_energy_5w/cache_fps",
    max_samples=1,
)

entry = dataset._raw[dataset.mol_ids[0]]
coords    = entry["electronic_density"]["coords"]
densities = entry["electronic_density"]["density"]

levels = extract_manifold_levels(coords, densities, n_levels=4)

for lv in levels:
    print(f"Level {lv.level_id} (c={lv.threshold:.4f}): {len(lv.components)} components")
    for comp in lv.components:
        print(f"  Component {comp.component_id}: {len(comp.verts)} verts, "
              f"scalar shape={comp.scalar_features.shape}, "
              f"vector shape={comp.vector_features.shape}")
```

---

## 缓存 I/O 辅助函数

以下三个函数在 `data/manifold.py` 末尾定义，供预处理脚本和后续 Stage 使用：

```python
def manifold_cache_path(cache_dir, mol_id, n_levels, smooth_sigma) -> str
def save_manifold_levels(path, levels) -> None
def load_manifold_levels(path) -> List[ManifoldLevel]
```

超参数（`n_levels`, `smooth_sigma`）被编码进缓存文件名，例如：
```
mol_001_nl4_s0.50.pkl
```
修改超参数后旧缓存文件不会被误读，会自动触发重新计算。

---

## 并行预处理脚本

`scripts/preprocess_stage1.py` 是批量预处理脚本，对全量 48k 分子运行 Stage 1 并将结果缓存到磁盘。

### 并行策略

- 使用 `multiprocessing.get_context("fork")` 创建进程池
- 主进程加载 9 GB PKL 后 fork worker，**共享内存（copy-on-write）**，worker 不会复制数据
- 各 worker 独立处理若干分子，结果写入独立缓存文件，无竞争
- 使用 `imap_unordered` 流式收集结果，支持实时进度输出

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pkl` | 必填 | PKL 文件路径 |
| `--cache-dir` | 必填 | 缓存目录 |
| `--workers` | `cpu_count/2` | 并行进程数 |
| `--n-levels` | 4 | 等值面层数 |
| `--percentiles` | `[20,40,60,80]` | 自定义百分位阈值 |
| `--smooth-sigma` | 0.5 | 高斯预平滑 σ（Bohr） |
| `--min-component-size` | 10 | 最小连通分量顶点数 |
| `--max-samples` | None | 处理分子数上限（测试用） |
| `--chunksize` | 4 | imap chunksize，较大值降低 IPC 开销 |
| `--no-merge` | False | 跳过合并步骤，保留独立 .pkl 文件 |

### 合并与清理

处理完成后默认执行合并步骤（`--no-merge` 可跳过）：

1. 顺序读取所有单分子 `.pkl` 文件（带 tqdm 进度条）
2. 合并为一个字典 `{mol_id: List[ManifoldLevel]}`，保存为：
   ```
   {cache_dir}/all_nl{n_levels}_s{smooth_sigma:.2f}.pkl
   ```
3. 删除所有单分子 `.pkl` 文件

> **注意：** 合并文件体积估计约为单分子平均大小 × 分子数，对 48k 分子通常在 5–20 GB 范围，请确保磁盘空间充足。

### 运行示例

```bash
# 小批量测试（20 个分子，4 个进程）
python scripts/preprocess_stage1.py \
    --pkl  data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --cache-dir data/ed_energy_5w/cache_manifold \
    --max-samples 20 --workers 4

# 全量预处理（建议在后台运行）
python scripts/preprocess_stage1.py \
    --pkl  data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --cache-dir data/ed_energy_5w/cache_manifold \
    --workers 8
```

### 进度条

使用 `tqdm` 显示进度条，ETA 基于墙钟时间的实际吞吐率（completed/sec）计算，**自动正确反映多进程并行加速**，不受单分子平均耗时影响。

进度条 postfix 实时显示：
```
Stage 1: 42%|████████▌           | 20160/47986 [08:12<11:23, 40.7mol/s]  ok=19840 cached=150 empty=12 err=3 avg=1.24s
```

`error` 和 `empty` 事件通过 `tqdm.write()` 打印到进度条上方，不会打断进度条显示。

### 加载已缓存结果

```python
from ed2e.data.manifold import load_manifold_levels, manifold_cache_path

path   = manifold_cache_path("data/ed_energy_5w/cache_manifold", mol_id, n_levels=4, smooth_sigma=0.5)
levels = load_manifold_levels(path)   # List[ManifoldLevel]
```

---

## 已知问题与修复

### RuntimeWarning: invalid value encountered in divide

**位置：** `compute_mesh_curvatures` 内的 `_cot` 辅助函数。

**原因：** `np.where(cond, A, B)` 在选择结果前会**无条件求值 A 和 B 两个分支**。当 `ssin = 0`（退化三角形）时，`dot / ssin` 仍被求值，触发除零警告。

**修复（已应用）：**
```python
# 修复前（触发警告）
return np.where(ssin > 1e-12, dot / ssin, 0.0)

# 修复后（无警告）
safe_sin = np.where(ssin > 1e-12, ssin, 1.0)   # 零替换为 1，避免求值时除零
return np.where(ssin > 1e-12, dot / safe_sin, 0.0)
```

---

## 注意事项

1. **体素网格重建**：PKL 数据已过滤至密度 > 0.05，重建时低密度区域体素填 0；Gaussian 平滑有助于消除该跳跃导致的等值面伪影。
2. **B-spline 系数共享**：同一分子的 `bspline_coeffs`（由 `spline_filter` 计算）在主函数中一次计算，传入所有层复用，避免重复计算。
3. **法向量方向约定**：所有法向量指向密度增大方向（即 ∇ρ > 0，分子核心方向），与论文 §A.1.2 的 $n(x) = \nabla\rho / \|\nabla\rho\|$ 一致。
4. **vector_features 坐标系**：存储在全局 3D Bohr 坐标系中；Stage 2 的 FCLC 构造将其投影到局部切坐标系。

---

## 可视化

`ed2e/utils/visualize_manifold.py` 提供等值流形的 3-D 可视化，支持两种后端。

### 后端

| 后端 | 输出 | 特点 |
|------|------|------|
| `plotly`（默认） | 交互式 HTML | 支持旋转/缩放，可保存为 `.html` |
| `matplotlib` | 静态图像 | 无额外重型依赖，可保存为 `.png` / `.pdf` |

两种后端均按层着色（内层红→暖色，外层蓝→冷色），外层透明度更高，内层叠加可见。原子坐标以球体形式叠加，按元素着色。

### CLI 用法

```bash
# 从合并 pkl 可视化一个分子（plotly，在浏览器中打开）
python ed2e/utils/visualize_manifold.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --mol-id <mol_id> \
    --atom-pkl data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl

# 保存为 HTML，不弹窗
python ed2e/utils/visualize_manifold.py \
    --manifold-pkl ... --mol-id <mol_id> \
    --save output.html --no-show

# matplotlib 后端，保存 PNG
python ed2e/utils/visualize_manifold.py \
    --manifold-pkl ... --mol-id <mol_id> \
    --backend matplotlib --save output.png

# 只渲染指定层（如只看外两层）
python ed2e/utils/visualize_manifold.py \
    --manifold-pkl ... --mol-id <mol_id> \
    --levels 2 3
```

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--manifold-pkl` | 合并 pkl 路径（dict）或单分子 pkl 路径（list） |
| `--mol-id` | 合并 pkl 中的分子 ID（单分子 pkl 时可省略） |
| `--atom-pkl` | 原始 PKL 路径，用于叠加原子坐标（可选） |
| `--levels` | 只渲染指定 level_id，如 `--levels 0 1`（默认全部） |
| `--backend` | `plotly`（默认）或 `matplotlib` |
| `--save` | 保存路径（`.html` / `.png` / `.pdf`） |
| `--no-show` | 不自动打开窗口（搭配 `--save` 使用） |

### 编程调用

```python
from ed2e.utils.visualize_manifold import visualize_manifolds
from ed2e.data.manifold import load_manifold_levels, manifold_cache_path

levels = load_manifold_levels(
    manifold_cache_path("data/ed_energy_5w/cache_manifold", mol_id, n_levels=4, smooth_sigma=0.5)
)
visualize_manifolds(
    levels,
    atom_coords=atom_coords,   # (N, 3) float32，可选
    atom_types=atom_types,     # (N,) int，可选
    backend="plotly",
    save_path="mol.html",
)
```
