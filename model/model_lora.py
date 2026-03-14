import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x)) 

'''把 LoRA 适配器动态挂到模型里的部分 nn.Linear 层上，
并改写这些层的 forward，让输出变成“原始线性层输出 + LoRA 输出”'''
def apply_lora(model, rank=8):
    # 遍历模型中的所有子模块
    for name, module in model.named_modules():
        # 只对方阵的线性层应用 LoRA
        # 即 weight 形状满足 [out_features, in_features] 且两者相等
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建对应尺寸的 LoRA 模块，并放到和模型相同的设备上
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(module.weight.device)

            # 把 lora 挂到当前线性层上，方便参数注册和后续访问
            setattr(module, "lora", lora)

            # 保存原始 forward，后面会在它的基础上叠加 LoRA 输出
            original_forward = module.forward

            # 用默认参数显式绑定当前循环里的 original_forward 和 lora，
            # 避免闭包引用到后续循环中被覆盖的变量
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            # 用新的 forward 替换原来的 forward
            module.forward = forward_with_lora


def load_lora(model, path):
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
