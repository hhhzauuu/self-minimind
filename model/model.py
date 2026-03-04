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

def precompute_freqs_cis(dim:int,end:int=int(32*1024),rope_base:float=1e6,rope_scaling:Optional[dict]=None):
    '''
    一次性预计算 RoPE（旋转位置编码）在每个位置、每个维度上的旋转角度对应的 cos / sin 表，
    后续在注意力里给 Q/K 做旋转时直接查表，避免每个 forward 都重复算三角函数

    dim: 每个注意力头的维度，因为RoPE是对每个注意力头独立计算的
    end: 预计算的最大位置，默认为32k
    rope_base: RoPE的基数，默认为1e6
    rope_scaling: 一个字典，RoPE缩放参数，如果不为None，则根据缩放参数调整预计算的频率
     - beta_fast: 快速位置编码的beta参数，默认为4
     - beta_slow: 慢速位置编码的beta参数，默认为1
     - factor: 快速位置编码和慢速位置编码的频率差异因子，默认为4
     - original_max_position_embeddings: 原始最大位置编码长度，默认为2048
     - type: 缩放类型，默认为"yarn"
    '''
    freqs,attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0 #torch.arange(0, dim, 2)生成一个从0到dim，步长为2的整数序列，表示偶数索引的位置；[: (dim // 2)]取前dim//2个元素，这是防御性编程。 .float()将整数序列转换为浮点数，然后除以dim
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = ( #orig_max是原始最大位置编码长度，factor是扩大倍数，beta_fast和beta_slow是高频和低频位置编码的beta参数，attn_factor是注意力缩放因子
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16), # 从字典里取值，取不到就默认值
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0: #如果推理长度大于训练长度，则需要调整频率
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base)) #这是一个反解函数，根据频率算出对应的维度索引。
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1) #low是低维度（高频）的索引的界限，也就是比这个频率还高的就不要修改；high是高维度（低频）的索引的界限，也就是比这个频率低就要修改；
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1) #平滑处理
            freqs = freqs * (1 - ramp + ramp / factor) #线性插值公式，高频不变，低频按照factor缩放，中频平滑处理
    #
    t = torch.arange(end, device=freqs.device) #生成位置索引，设备在gpu还是cpu无所谓，只跟着freqs就行。
    freqs = torch.outer(t, freqs).float() #外积，生成一个[end, dim//2]的矩阵，每行是一个位置，每列是一个频率（因为每两个数共享一个频率）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor #对频率矩阵进行复制，因为频率只有dim//2个，但是位置有dim个。
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin #返回预计算的cos和sin表，形状都是[end, dim]
 
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    '''
    将预计算好的 RoPE cos/sin 表应用到 Q 和 K 上，完成旋转位置编码的注入。
    '''
    def rotate_half(x): #把后半部分取负放到前面，前半部分原样放到后面。 这其实就是二维旋转矩阵里"正交分量"的排列方式
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # unsqueeze是为了中间插入一个维度 1
    # Q 形状是 [Batch, Seq, Heads, Dim]，cos/sin 形状是 [Seq, Dim]，unsqueeze_dim=1 就是在 Seq 和 Dim 之间插入一个维度，变成 [Seq, 1, Dim]
    # 广播是靠右对齐的，所以在乘法运算时，cos/sin 会自动广播到 [Seq, Heads, Dim] 的形状，和 Q/K 的形状匹配
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed
