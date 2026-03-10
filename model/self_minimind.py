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
        # self.flash_attention = flash_attention
        self.flash_attn = flash_attention
        # self.use_moe = use_moe
        # self.num_experts_per_tok = num_experts_per_tok
        # self.n_routed_experts = n_routed_experts
        # self.n_shared_experts = n_shared_experts
        # self.seq_aux = seq_aux
        # self.norm_topk_prob = norm_topk_prob
        # self.aux_loss_alpha = aux_loss_alpha
        # self.scoring_func = scoring_func
        self.use_moe = False
        self.num_experts_per_tok = 0
        self.n_routed_experts = 0
        self.n_shared_experts = 0
        self.seq_aux = False
        self.norm_topk_prob = False
        self.aux_loss_alpha = 0.0
        self.scoring_func = "softmax"

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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    因为GQA中K和V的数量比Q少，所以需要把K和V重复n_rep倍来匹配Q的数量，方便后续的矩阵运算。
    """
    #输入的形状为[batch_size, seq_length, num_key_value_heads, head_dim]。其中num_key_value_heads是K/V头的数量。
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: #如果不需要重复，直接返回原始的K/V张量
        return x
    return (
        # x[:, :, :, None, :]在num_key_value_heads和head_dim之间插入一个维度
        # expand(bs, slen, num_key_value_heads, n_rep, head_dim)将这个新维度扩展为n_rep,这样就得到了一个形状为[batch_size, seq_length, num_key_value_heads, n_rep, head_dim]的张量.
        # reshape(bs, slen, num_key_value_heads * n_rep, head_dim)将num_key_value_heads和n_rep这两个维度合并成一个维度
        # num_key_value_heads * n_rep 相当于实现了K/V的重复，但不需要实际复制数据，节省内存和计算资源。
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self,args: SelfMiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads#
        assert args.num_attention_heads % self.num_key_value_heads == 0 #Q的数量必须是K/V数量的整数倍。
        self.n_local_heads = args.num_attention_heads #Q头的数量，
        self.n_local_kv_heads = self.num_key_value_heads#K/V头的数量，
        self.n_rep = self.n_local_heads // self.n_local_kv_heads#KV需要重复的次数
        self.head_dim = args.hidden_size // args.num_attention_heads#每个注意力头Q的维度,也就是hidden_size除以Q的个数
        # 初始化QKV,注意KV的输出维度是num_key_value_heads * head_dim，跟Q不一样。
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 最后的投影层，输入维度是num_attention_heads * head_dim，输出维度是hidden_size，这样才能和残差连接的输入维度匹配。
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        # 注意力的dropout层和残差连接的dropout层
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        #是否使用Flash Attention的标志，加速计算。
        flash_enabled = getattr(args, 'flash_attention', getattr(args, 'flash_attn', False))
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and flash_enabled
    
    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin表
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,#kvcache
                use_cache=False,#要不要把当前的K/V放到cache里
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape #这个时候输入的形状为[batch_size, seq_length, hidden_size]
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x) #分别通过线性层得到Q、K、V的形状。
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 从输入的position_embeddings中解包出预计算的cos和sin表，形状都是[seq_len, head_dim]，position_embeddings具体怎么获取的是在外层model上实现的,不是attention层考虑的。
        cos, sin = position_embeddings 
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin) #经过RoPE得到带有位置信息的QK

        # kv_cache实现
        if past_key_value is not None: # 如果传入了历史缓存，说明正在逐字生成，需要把当前的KV和历史的拼接起来。
            xk = torch.cat([past_key_value[0], xk], dim=1)# 在dim=1的维度上拼接，也就是seq_len的维度.
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None #如果开关是打开的，就把新的KV放到cache

        # 调整qkv的维度，并对kv进行复制
        # xq的形状是[batch_size, seq_length, num_attention_heads, head_dim]，需要转置成[batch_size, num_attention_heads, seq_length, head_dim]，因为后续的运算是对每个注意力头独立进行的。
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 当满足使用Flash Attention的条件时，且输入序列长度大于1，且没有历史KV缓存，且没有掩码。才能用Flash Attention。
        # 大模型在训练时，不是逐字生成的，而是整段文字的并行生成。所以就不需要KV cache。
        # 这么就容易出现显存爆炸问题，而Flash Attention可以降低显存占用。
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            '''
            此时的张量形状为：
            xq: [batch_size, num_attention_heads, seq_length, head_dim],记为[B, H, L, D]
            PyTorch 会自动保留前两个维度（B和H）作为批处理并行计算，只对最后两个维度做矩阵乘法
            '''
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) # （L,D) @ (D,L) -> (L,L)。scores的形状是[B,H,L,L]
            # 这里是做的因果掩码 causal mask。因为这是自回归模型
            # torch.full((seq_len, seq_len), float("-inf"))：先造一个L*L的矩阵，里面全是负无穷
            # torch.triu(..., diagonal=1)：triu 意思是 Upper Triangle（上三角）。它把刚刚那个矩阵的下半部分变成0，只保留对角线及右上方的负无穷；diagonal=1表示 0往上移一行，也就是对角线也是0
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
            
            # 这里是padding掩码，因为输入的序列可能有padding部分，这些部分不应该对注意力计算产生影响，所以要把它们的分数设置为负无穷。 
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 拼接多个注意力头
        # output的形状是[B,H,L,D]，需要转置回[B,L,H,D]，然后再reshape成[B,L,H*D]，也就是[B,L,hidden_size]，这样才能和残差连接的输入维度匹配。
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output)) #先通过线性层投影,然后再dropout
        return output, past_kv #返回输出和新的KV缓存

class FeedForward(nn.Module):
    def __init__(self, config: SelfMiniMindConfig):
        super().__init__()
        
        # intermediate_size是FFN中间层的维度。
        # 在传统transformer中，中间层一般是隐藏层的4倍。为了保证总参数量不变，在swiglu中，intermediate_size=hidden_size * 8/3
        # 为了提高gpu并行效率，intermediate_size需要是64的倍数，所以这里做了一个调整，向上取整到最近的64的倍数。
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)# 右侧门控线性层
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)# 降维线性层
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False) # 左侧升维线性层
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act] #利用字典根据配置文件的字符串名称，找到对应的激活函数

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class SelfMiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: SelfMiniMindConfig):#layer_id是当前块在整个模型的第几块
        super().__init__()
        self.num_attention_heads = config.num_attention_heads # Q的数量，它其实是为了下面计算head_dim准备的
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config) # 算出来注意力

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 这里指的是GQA模块前的RMSNorm，也就是pre-norm，并没有在模块内实现。
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) #GQA模块后的RMSNorm,也就是FFN模块前的RMSNorm
        # self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        '''
        hidden_states:当前批次的所有词向量数据，形状为[batch_size, seq_len, hidden_size
        position_embeddings：预计算的RoPE位置编码表，包含cos和sin两部分，形状都是[seq_len, head_dim]
        '''
        residual = hidden_states # 备份，为做残差连接做准备
        hidden_states, present_key_value = self.self_attn( #经过注意力层(pre-norm)，得到新的hidden_states和新的KV缓存
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual # 残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)) # pre-norm -> FFN -> 残差连接
        return hidden_states, present_key_value #返回当前块的输出和新的KV缓存，供下一块使用
    

class SelfMiniMindModel(nn.Module):
    def __init__(self, config: SelfMiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers #vocab_size是词表大小，num_hidden_layers是transformer块的数量
        
        # 注意这里的Embedding,里面不是BGE这种Embedding模型，而是随机的权重矩阵，需要大模型的训练。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) #词嵌入层，输入是token id（是对输入的文本分词后的索引）形状为[batch_size, seq_len]，输出是对应的词向量，形状为[batch_size, seq_len, hidden_size]
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([SelfMiniMindBlock(l, config) for l in range(self.num_hidden_layers)]) #创建了一个长度为 num_hidden_layers 的列表，每个元素都是一个 SelfMiniMindBlock
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) #每个block都有pre-norm，等所有的block结束后，还有一个全局的RMSNorm，作用是对整个模型的输出做归一化处理，稳定训练。

        # 生成 RoPE 位置编码表，每个注意力头独立计算，所以维度是hidden_size // num_attention_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,#end是预计算的最大位置，rope_base是RoPE的基数
                                                    rope_scaling=config.rope_scaling)#是否启用长上下文的 RoPE 缩放策略,yarn
        self.register_buffer("freqs_cos", freqs_cos, persistent=False) #把 cos 表注册成 buffer，buffer 的含义是：它属于模型,会随着 model.to(device) 自动搬到 GPU；但它不是可训练参数，不参与梯度更新。
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,#输入的token id，形状为[batch_size, seq_len]
                attention_mask: Optional[torch.Tensor] = None,#padding掩码，形状为[batch_size, seq_len】
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,#历史KVcache列表，列表长度为transformer块的数量，每个元素是一个元组，包含K和V的张量，形状分别为[batch_size, seq_len,num_key_value_heads, head_dim]
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None #如果传进来的不是列表，而是layers对象，就视作none
        past_key_values = past_key_values or [None] * len(self.layers)

        # 看列表第一层的KV缓存，如果不空，说明正在逐字生成，那么start_pos就等于历史缓存的长度，也就是已经生成的序列长度；如果空，说明是并行生成，start_pos就从0开始。
        # 已经生成的序列长度怎么看？通过past_key_values的第一层，也就是past_key_values[0]，找到这个元组的第一个元素（也就是K)的长度seq_len就好
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        
        # 初始的特征，也就是输入transformer块的词向量。
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = ( # 从预计算好的表里面，取出来和输入序列匹配的部分。
            # 同一个position只和当前位置有关系，所以会被所有block共享
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = [] #存放新的kvcache
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)): # 遍历每个transformer块，调用它的forward方法。
            hidden_states, present = layer( #调用了block的forward方法！
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present) #更新kvcache

        hidden_states = self.norm(hidden_states)# 最后一个全局的rmsnorm

        # aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        # return hidden_states, presents, aux_loss
        return hidden_states, presents #注意这里要返回的不只是输出的hidden_states，还有新的KV缓存presents! 
    
class SelfMiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SelfMiniMindConfig

    def __init__(self, config: SelfMiniMindConfig = None):
        self.config = config or SelfMiniMindConfig()
        super().__init__(self.config)
        self.model = SelfMiniMindModel(self.config) #主干网络，输出的是隐藏状态hidden_states
        
        # 语言模型头，就是把隐藏状态映射到词表大小的线性层。[B,L,H]->[B,L,V]
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享。为什么不需要转置？embed_tokens的形状为[vocab_size, hidden_size]，lm_head的权重形状为[hidden_size, vocab_size]
        # 但是在pytorch中，线性层的权重是[out_features, in_features]，也就是[vocab_size, hidden_size]，所以它们的形状是一样的，可以直接赋值共享权重。
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, # 训练标签，形状为[batch_size, seq_len]
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0, #保留哪些logits
                **args):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )

        # 取出要保留的logits部分
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]) #[B,L',H] -> [B,L',V]

        loss = None # 推理时没有损失
        if labels is not None: 
            '''
            因为自回归语言模型的训练目标是预测下一个词，所以在计算交叉熵损失时，输入的logits需要去掉最后一个时间步（因为它没有对应的标签），而labels需要去掉第一个时间步（因为它没有对应的输入）。这样就实现了对齐。
            '''
            shift_logits = logits[..., :-1, :].contiguous() #contiguous()的作用是确保在内存中是连续存储的，这样才能正确地调用view()方法进行reshape。
            shift_labels = labels[..., 1:].contiguous()

            # 计算交叉熵损失
            # shift_logits的形状是[B, L-1, V]-> [B*(L-1), V]，shift_labels的形状是[B, L-1] -> [B*(L-1)]，这样就满足了交叉熵损失函数的输入要求。
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        # 把输出打包，符合huggingface接口规范
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        # output.aux_loss = aux_loss
        return output