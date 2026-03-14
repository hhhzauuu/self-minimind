import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.self_minimind import SelfMiniMindConfig
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # 【区别1：学习率调度】LoRA微调不需要warmup（预热步长），因为基座模型是稳定的；而预训练通常需要传入 warmup_steps
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # self_minimind may not expose aux_loss; keep backward compatible.
            aux_loss = getattr(res, "aux_loss", None)
            loss = res.loss if aux_loss is None else (res.loss + aux_loss)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 【区别2：梯度裁剪范围】预训练是裁剪整个模型的梯度 clip_grad_norm_(model.parameters())
            # LoRA只对参与训练的旁路矩阵参数 lora_params 进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item() if aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, '
                f'epoch_time: {eta_min:.1f}min'
            )
            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "logits_loss": current_logits_loss,
                        "aux_loss": current_aux_loss,
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    }
                )

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            # 【区别3：模型保存】预训练保存的是包含几百兆甚至几G的全部 `state_dict` 权重
            # LoRA微调只需要提取并保存lora层相关的权重即可，存下的 pth 体积通常只有十几兆
            save_lora(model, lora_save_path)
            lm_checkpoint(
                lm_config,
                weight=args.lora_name,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints',
            )
            model.train()

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SelfMiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out/lora", help="model output dir")
    parser.add_argument("--lora_name", type=str, default="lora_identity", help="adapter weight name")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="initial lr")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="mixed precision type")
    parser.add_argument("--num_workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="grad accumulation")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="grad clip")
    parser.add_argument("--log_interval", type=int, default=10, help="log interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="save interval")
    parser.add_argument('--hidden_size', default=512, type=int, help="hidden size")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="num layers")
    parser.add_argument('--max_seq_len', default=340, type=int, help="max seq len")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="kept for compatibility")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_identity.jsonl", help="SFT data path for LoRA")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="base checkpoint name")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="resume from checkpoint")
    parser.add_argument("--use_wandb", action="store_true", help="enable swanlab")
    parser.add_argument("--wandb_project", type=str, default="SelfMiniMind-LoRA", help="swanlab project")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="torch.compile")
    args = parser.parse_args()

    # 1) Distributed init + seed
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 2) Config + resume checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    if args.use_moe:
        Logger('MoE is disabled in current self_minimind; using dense FFN.')
    # Fix #1: use current config class from self_minimind.
    lm_config = SelfMiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume == 1 else None

    # 3) AMP context
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # 4) Logger setup
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = (
            f"SelfMiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-"
            f"BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5) Key differences vs train_pretrain:
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        
    # 【区别4：模型结构修改】LoRA需要调用 apply_lora 将低秩旁路矩阵注入到原模型的 Linear 层旁边
    apply_lora(model)

    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM total params: {total_params / 1e6:.3f} M")
    Logger(f"LoRA params: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA ratio: {lora_params_count / total_params * 100:.2f}%")

    # 【区别5：参数冻结】遍历模型参数，只将名字包含 'lora' 的参数设为 requires_grad = True
    # 基座模型的原始参数需全部冻结计算（requires_grad = False）
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    # 【区别6：数据集不同】LoRA微调针对的是对话类对齐训练，使用带标签结构的 SFTDataset
    # 而预训练使用的是只包含长文本首尾相连的无监督 PretrainDataset
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 【区别7：优化器控制范围不同】预训练交给优化器的是全模型参数：model.parameters()
    # LoRA仅将单独提取出的小规模 lora_params 交给优化器进行正反向计算
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    # 6) Resume state
    start_epoch, start_step = 0, 0
    if ckp_data:
        # strict=False is intentional: adapter keys can vary across runs/versions.
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # 7) DDP wrap
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 8) Train
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: skip first {start_step} steps, resume at step {start_step + 1}')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)

    # 9) Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
