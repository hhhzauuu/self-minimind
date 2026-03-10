# self-minimind

从0到1学习大模型的极简实现，当前代码聚焦在核心 Transformer 组件复现与预训练链路打通。

## 环境

项目推荐使用 `uv` 管理依赖和虚拟环境：

```bash
uv sync
```

Windows：

```powershell
.venv\Scripts\activate
```

Linux / macOS：

```bash
source .venv/bin/activate
```

如果缺少额外依赖，可以直接补装：

```bash
uv add datasets transformers torch swanlab
```

## 代码结构

- [model/self_minimind.py](model/self_minimind.py)：模型主干与配置，包含 `SelfMiniMindConfig`、RMSNorm、RoPE/YaRN、GQA、SwiGLU 风格 FFN、`SelfMiniMindModel`、`SelfMiniMindForCausalLM`。
- [dataset/lm_dataset.py](dataset/lm_dataset.py)：预训练数据集定义，把 `{"text": ...}` 样本转成 `input_ids` 与 `labels`。
- [trainer/train_pretrain.py](trainer/train_pretrain.py)：预训练入口，支持 AMP、梯度累积、DDP、断点续训和 SwanLab 日志记录。
- [trainer/trainer_utils.py](trainer/trainer_utils.py)：学习率调度、模型初始化、checkpoint、分布式辅助函数。
- [doc/学习日志.md](doc/学习日志.md)：更完整的公式推导、代码笔记和训练流程记录。

## 模型概览

整体数据流如下：

`input_ids -> nn.Embedding -> N x SelfMiniMindBlock -> RMSNorm -> lm_head -> logits`

可以把当前模型理解成两层：

- `SelfMiniMindModel`：负责主干网络计算。它把 `input_ids` 先映射成词向量，然后依次通过多层 `SelfMiniMindBlock`，最后输出上下文化后的 `hidden_states`。
- `SelfMiniMindForCausalLM`：在主干外再接一个 `lm_head`，把 `hidden_states` 投影到词表维度，得到 `logits`，并在训练时计算 next-token prediction 的 loss。

单个 `SelfMiniMindBlock` 采用的是比较典型的 Pre-Norm 结构：先做 `RMSNorm`，再进入 GQA Attention，接着做残差连接；随后再经过一次 `RMSNorm`、SwiGLU 风格 FFN 和第二次残差连接。推理阶段还支持 `past_key_values`，也就是常见的 KV Cache。

当前实现的关键点：

- Pre-Norm Transformer Block
- RoPE 位置编码，并预留 YaRN 长上下文外推能力
- GQA 注意力机制
- SwiGLU 风格 FFN
- `embed_tokens.weight` 与 `lm_head.weight` 权重共享

## 预训练快速开始

1. 将 `pretrain_hq.jsonl` 放到仓库根目录的 `dataset/` 下。
2. 确保 `model/` 目录下至少有 `tokenizer_config.json`；如果同时有 `tokenizer.json`，分词会更快。
3. 如果要记录训练日志，先执行 `swanlab login`。
4. 进入 `trainer/` 目录再启动训练。当前脚本默认使用 `../model`、`../dataset`、`../out` 等相对路径，在 `trainer/` 目录下执行最稳妥。

单卡示例：

```powershell
Set-Location .\trainer
python train_pretrain.py --data_path ../dataset/pretrain_hq.jsonl --use_wandb
```

多卡示例：

```powershell
Set-Location .\trainer
torchrun --nproc_per_node=2 train_pretrain.py --data_path ../dataset/pretrain_hq.jsonl
```

补充说明：

- 当前脚本默认使用 `bfloat16` 自动混合精度。
- `--use_moe` 目前仅保留为兼容参数，当前训练固定使用普通 FFN，不启用 MoE。

## 学习资料

如果想看更详细的推导、代码逐段解释和工程实践记录，可以直接阅读 [doc/学习日志.md](doc/学习日志.md)。
