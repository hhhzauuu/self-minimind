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

**为什么需要归一化？**
在深度学习模型的反向传播过程中，梯度的值往往与正向阶段的输入数据 $x$ 有关。当 $x$ 的绝对值过大或者过小，就容易引发**梯度爆炸**或**梯度消失**。因此，需要引入归一化（Normalization）操作，将数据特征放缩到一个稳定的尺度范围内（即让标准差变为 1），从而确保训练平稳收敛。

**为什么选择 RMSNorm？**
传统 Transformer 常用的是 LayerNorm（层归一化），而在这里被替换成了 RMSNorm（均方根归一化）。相比与 LayerNorm，**RMSNorm 直接舍弃了对均值的计算和去均值（中心化）的步骤**。在大规模模型的训练中，这样做可以极其显著地降低计算量、提升效率，而且其实际效果通常与 LayerNorm 不相上下甚至更好。

**数学公式：**
$$y_i = \frac{x_i}{\text{RMS}(x)} * \gamma_i$$
其中：
$$\text{RMS}(x) = \sqrt{\epsilon + \frac{1}{n} \sum_{i=1}^{n} x_i^2}$$
*($\gamma_i$ 即代码中可学习的参数 `weight`，$\epsilon$ 是为了防止分母为0而设定的微小实数)*

**代码实现：**
```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # 输入的特征维度
        self.hidden_size = hidden_size
        # 防止除 0 的 epsilon 参数稳定性项
        self.eps = eps
        # 可学习的权重参数，初始化为全1，维度与特征维度保持一致
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, x):
        # 输入格式通常为 [batch_size, seq_length, hidden_size]
        # .pow(2) 表示求平方；.mean(-1) 表示在最后一个维度(按 hidden_size 跨度)求均值。
        # keepdim=True 保持截断前的张量维度不变，确保最后输出能够通过广播机制扩展（形状如 [batch_size, seq_length, 1]）。
        variance = x.pow(2).mean(-1, keepdim=True)
        
        # 计算均方根倒数对输入直接相乘。
        # torch.rsqrt 相当于计算 1.0 / torch.sqrt(variance + self.eps)，底层 C++ 计算更高效。
        x = x * torch.rsqrt(variance + self.eps)
        
        # 乘以 RMSNorm 本身可学习的一维缩放权重
        return self.weight * x
```