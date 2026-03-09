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

## 快速理解

`self-minimind` 是一个面向学习的大模型极简实现，主干代码集中在 [model/model.py](model/model.py)。

整体数据流可以概括为：

`input_ids -> nn.Embedding -> N x SelfMiniMindBlock -> RMSNorm -> lm_head -> logits`

其中：

- `SelfMiniMindBlock`：由 `GQA Attention + FFN(SwiGLU)` 组成，采用 Pre-Norm 和残差连接。
- `RoPE / YaRN`：负责位置编码和长上下文外推。
- `KV Cache`：用于推理阶段复用历史 K/V，加速逐 token 生成。

## 核心组件

### 1. RMSNorm

使用 RMSNorm 代替 LayerNorm，减少计算量并提升大模型训练与推理效率。

### 2. RoPE + YaRN

- `precompute_freqs_cis(...)`：预计算 RoPE 所需的 `cos/sin` 表。
- `apply_rotary_pos_emb(...)`：把位置编码应用到 Q/K。
- `inference_rope_scaling=True` 时启用 YaRN 风格的频率缩放，用于更长上下文推理。

### 3. GQA Attention

注意力模块采用 GQA（Grouped Query Attention）：Q 头数多于 K/V 头数，通过 `repeat_kv(...)` 把较少的 K/V 头扩展到与 Q 匹配。

实现上同时支持：

- 因果掩码与 padding 掩码
- 推理用的 `past_key_values`
- 条件满足时走 `scaled_dot_product_attention` 路径

### 4. FFN (SwiGLU)

FFN 采用 SwiGLU 风格结构：

- `gate_proj` 生成门控信号
- `up_proj` 提供候选特征
- 两路特征逐元素相乘后，再通过 `down_proj` 投回 `hidden_size`

`intermediate_size` 默认取约 $\frac{8}{3} \cdot hidden\_size$，这是为了在使用三层线性映射时保持参数量大致稳定。

## 模型结构

### SelfMiniMindBlock

每个 Block 的计算顺序是：

`RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual`

对应公式：

$$
h_1 = x + Attention(RMSNorm(x))
$$

$$
h_2 = h_1 + FFN(RMSNorm(h_1))
$$

### SelfMiniMindModel

`SelfMiniMindModel` 是主干网络，只负责输出上下文化后的隐藏状态，不直接输出词表概率。

当前接口与 [model/model.py](model/model.py) 保持一致：

```python
hidden_states, presents = model(
    input_ids,
    attention_mask=None,
    past_key_values=None,
    use_cache=False,
)
```

返回值说明：

- `hidden_states`：最后一层的隐藏表示
- `presents`：每层更新后的 KV Cache

### SelfMiniMindForCausalLM

`SelfMiniMindForCausalLM` 在主干外加上语言模型头 `lm_head`，负责把 `[B, L, H]` 映射到 `[B, L, V]`，并在训练时计算 next-token prediction 的交叉熵损失。

当前接口与 [model/model.py](model/model.py) 保持一致：

```python
output = model(
    input_ids,
    attention_mask=None,
    labels=None,
    past_key_values=None,
    use_cache=False,
    logits_to_keep=0,
)
```

其中：

- `labels` 不为 `None` 时计算 loss
- `logits_to_keep` 用于只保留部分位置的 logits
- 返回值是 `CausalLMOutputWithPast`

另外，`embed_tokens.weight` 与 `lm_head.weight` 做了权重共享，以减少参数量并保持输入嵌入与输出投影的一致性。

## 更多说明

如果你想看更详细的推导、公式说明和逐段代码笔记，可以直接看 [doc/学习日志.md](doc/学习日志.md)。
