from transformers import PretrainedConfig


class SelfMiniMindConfig(PretrainedConfig):
    model_type = "self-minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1, #序列起始标记在词表中的id，默认为1
        eos_token_id: int = 2, #序列结束标记在词表中的id，默认为2
        hidden_act: str = "silu", #激活函数
        hidden_size: int = 512, #隐藏层大小，默认为512
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8, #注意力头(Query)的数量
        num_hidden_layers: int = 8, #transformer块的数量
        num_key_value_heads: int = 2, #K，V的个数，比Q少，说明用的是GQA
        vocab_size: int = 6400, #词表大小，默认为6400
        rms_norm_eps: float = 1e-05, #RMSNorm的epsilon参数，默认为1e-05
        rope_theta: int = 1000000, #旋转位置编码的theta参数，默认为1000000
        inference_rope_scaling: bool = False, #是否在推理时使用ROPE缩放，默认为False
        flash_attention: bool = True, #是否使用Flash Attention，默认为True
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size #输入的特征维度
        self.eps = eps #防止除0的epsilon参数
        self.weight = nn.Parameter(torch.ones(hidden_size)) #可学习的权重参数，初始化为全1，维度为输入特征维度
    def forward(self,x):
        variance = x.pow(2).mean(-1, keepdim=True) # 输入的格式为[batch_size, seq_length, hidden_size]; -1 表示对最后一个维度进行求均值，即对hidden_size维度求均值；keepdim=True保持维度不变，输出的格式为[batch_size, seq_length, 1]
        x = x * torch.rsqrt(variance + self.eps) #等价于 x / torch.sqrt(variance + self.eps)，对输入进行归一化处理，除以标准差
        return self.weight * x #乘以可学习的权重参数