# self-minimind
从0到1学习大模型。

## 环境配置

本项目推荐使用 `uv` 进行依赖和虚拟环境管理。请按照以下步骤配置开发环境：

### 1. 安装 uv
如果你还没有安装 `uv`，可以通过 pip 进行安装：
```bash
pip install uv
```

### 2. 初始化 uv 环境
初始化项目并指定使用的 Python 版本：
```bash
uv init -p 3.12
```

### 3. 修改依赖配置（注意保存）
在 `pyproject.toml` 文件中修改 `dependencies` 字段以添加你需要的依赖包。
**修改完毕后，请务必保存文件！！！**

### 4. 安装/同步环境
运行以下命令，`uv` 会根据配置文件自动下载第三方库并创建 `.venv` 虚拟环境：
```bash
uv sync
```
*(更推荐直接使用 `uv add package_name`，它会自动修改配置文件、安装依赖并更新环境文件。)*

### 5. 激活环境
配置完成后，激活虚拟环境开始开发：
```powershell
.venv\Scripts\activate
```
*(如果是 Linux / macOS 系统，请使用 `source .venv/bin/activate`)*

## 架构实现

### 1. 均方根归一化 (RMSNorm)
**代码实现：**
```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__()
        self.dim = dim #输入的特征维度
        self.eps = eps #防止除0的epsilon参数
        self.weight = nn.Parameter(torch.ones(dim)) #可学习的权重参数，初始化为全1，维度为输入特征维度
    def _norm(self,x):
        # 输入的格式为[batch_size, seq_length, hidden_size]; -1 表示对最后一个维度进行求均值，即对hidden_size维度求均值；keepdim=True保持维度不变，输出的格式为[batch_size, seq_length, 1]
        #等价于 x / torch.sqrt(variance + self.eps)，对输入进行归一化处理，除以标准差
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
    def forward(self,x):
        return self.weight * self._norm(x.float()).type_as(x) #乘以可学习的权重参数，.float()将输入转为32位浮点数，.type_as(x)将输出的类型转换回输入的类型

```

### 2. 旋转位置编码 (RoPE)

RoPE（Rotary Position Embedding）把“位置信息”变成对向量的**二维旋转**，并且只作用在 **Q / K** 上。
这样在计算注意力相似度时，会自然地体现相对位移（更利于长文本建模）。

本项目在 [model/model.py](model/model.py) 中实现了两步：

- `precompute_freqs_cis(...)`：一次性预计算 `cos/sin` 查表（形状约为 `[max_seq_len, head_dim]`），避免每次 forward 重复计算三角函数
- `apply_rotary_pos_emb(...)`：把 `cos/sin` 应用到 Q/K（采用“对半配对”的旋转实现）

最小用法示意：
```python
freqs_cos, freqs_sin = precompute_freqs_cis(
    dim=head_dim,
    end=max_seq_len,
    rope_base=rope_theta,
    rope_scaling=rope_scaling,
)

# 按当前序列长度截取 cos/sin（增量推理时通常按 position 切片）
q, k = apply_rotary_pos_emb(q, k, freqs_cos[:seq_len], freqs_sin[:seq_len])
```

### 3. YaRN（长上下文外推）

当推理长度超过训练时的最大长度，原始 RoPE 可能出现外推不稳。YaRN 的核心思路是：

- 高频维度保持不变（更关注局部信息）
- 低频维度做缩放/内插（让旋转变慢以容纳更长上下文）
- 中频维度用 ramp 做平滑过渡，并可通过 `attention_factor` 做注意力熵修正（代码里对应 `attn_factor`）

在配置上：`SelfMiniMindConfig.inference_rope_scaling=True` 会生成 `rope_scaling` 字典，并在 `precompute_freqs_cis(...)` 中生效。

### 4. GQA (Grouped Query Attention) 与注意力计算

GQA 是一种介于 MHA (多头注意力) 和 MQA (多查询注意力) 之间的优化方案：多个 Query (Q) 头共享同一组 Key (K) 和 Value (V) 头，在保持效果的同时大幅降低显存占用并提升推理速度。

在 [model/model.py](model/model.py) 中的 `Attention` 模块融合了以下关键技术：

- **张量重复计算：** 通过 `repeat_kv` 将较少头的 K、V 张量进行扩展补齐，以满足矩阵运算规则。
- **并行与 KV Cache：** 推理阶段通过返回并传入 `past_kv`（KV Cache）来加速逐测生成；训练阶段则无需缓存，通过并行即可获取所有结果。
- **Flash Attention 支持：** 当处于训练模式且张量长度允许时，启用加速计算模式，降低大规模矩阵导致显存爆炸的风险。
- **双重掩码机制 (Mask)：**
  - **因果掩码 (Causal Mask)：** 构建上三角掩码矩阵并在加到对应得分上（被设为 `-inf`），防自回归模型偷看"未来"词。
  - **填充掩码 (Padding Mask)：** 针对无实义的 Padding token 给定趋于负无穷的手动偏移分数值（如 `-1e9`），确保注意力绝不被无效位置吸引。
- **收尾投影：** 计算出多头结果后，经过维度转置、特征拼接展平，最后通过 `o_proj` 层混合所有注意力头进行特征投影输出。
