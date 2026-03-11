# self-minimind

从 0 到 1 学习大模型的极简实现路径。
小白如何学会并使用 minimind？本项目将为你提供答案。

- 参考项目：[minimind](https://github.com/jingyaogong/minimind)
- 主要参考视频：[Bilibili: Only三小时！Pytorch从零手敲大模型，架构到训练全教程](https://www.bilibili.com/video/BV1T2k6BaEeC/?share_source=copy_web&vd_source=2fca6e2667a11ddcb7554d3ef302d0ff)

本项目致力于提供一条清晰、极简的大语言模型（LLM）学习与落地路径。项目深度参考了优秀的开源工程 MiniMind、b站相关教学视频。

与单纯的代码复现不同，本项目最大的特色在于系统性地补齐了原项目背后的底层基础知识（涵盖 Transformer 架构细节、RoPE 位置编码、KV Cache 机制以及 SwiGLU 等核心组件）。旨在帮助学习者跨越理论壁垒，不仅能把代码“跑通”，更能真正透彻掌握模型原理，从底层逻辑出发，亲手搭建出完全属于自己的大模型。

如果你想看更详细的推导、代码逐段解释和工程实践记录，可以直接阅读：[doc/学习日志.md](doc/学习日志.md)。

另外，如果想看简单易懂的 Hot 100 题解，请看：[doc/hot100刷题.md](doc/hot100刷题.md)。

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

整体数据流与架构可以参考下图：

<!-- 架构图来自学习日志 -->
![](https://cdn.nlark.com/yuque/0/2026/png/51029207/1772507093889-fb4e973c-c453-4259-84fb-bb6039de658e.png)

整体数据流如下：

`input_ids -> nn.Embedding -> N x SelfMiniMindBlock -> RMSNorm -> lm_head -> logits`

本项目模型结构可以从宏观上理解为两层封装：

- `SelfMiniMindModel`：负责主干网络计算。它将输入的单词索引 `input_ids` 映射成稠密的词向量（Embedding），随后依次通过 $K$ 层堆叠的 Transformer 层（即 `SelfMiniMindBlock`）。它的核心作用是提炼并融合上下文信息，最后输出上下文感知后的向量表示 `hidden_states`。
- `SelfMiniMindForCausalLM`：这是自回归语言模型（Causal LM）的头部封装。在执行完主干网络的计算后，利用线性映射层 `lm_head` 将提取出的 `hidden_states` 映射回完整的词表维度之上，从而得到每个可能词汇被生成的预测得分（`logits`）。在训练阶段，模型会据此和真实标签计算 Next-Token Prediction（预测下一个词）的交叉熵损失。

而在微观层面，单个 `SelfMiniMindBlock` 采用了目前业界主流的前置归一化（Pre-Norm）结构：
1. **输入阶段**：数据首先经历 `RMSNorm` 均方根归一化。
2. **注意力机制**：经过归一化后的隐状态进入 **GQA (Grouped Query Attention)** 进行分组多头注意力计算。该过程结合了 RoPE 旋转位置编码，之后利用残差连接和原始输入相加。
3. **前馈神经网络**：随后再度进行一次 `RMSNorm` 处理，输入包含门控机制的 SwiGLU 风格前馈神经网络（FFN），最后进行第二次残差连接。

*(推理阶段，模型内部引入了 `past_key_values`，完整支持 KV Cache 的存取，以大幅加速单步文本生成效率。)*

当前实现的关键点：

- Pre-Norm Transformer Block
- RoPE 旋转位置编码，预留 YaRN 长上下文外推能力
- GQA 分组查询注意力机制（在计算开销与效果间取得更佳平衡）
- 基于 SiLU 的 SwiGLU 风格可学习门控 FFN
- `embed_tokens.weight` 与 `lm_head.weight` 词表权重共享

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

