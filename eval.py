import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.self_minimind import SelfMiniMindConfig, SelfMiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    # 先从指定路径加载分词器 (Tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from: #如果是导入的是我们预训练用的tokenizer,然后用我们预训练模型权重
        model = SelfMiniMindForCausalLM(SelfMiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            # use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        # moe_suffix = '_moe' if args.use_moe else ''
        # ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'

        # 拼接底座模型路径：如果权重名包含sft则在sft目录下，包含pretrain则在pretrain目录下
        task_dir = 'sft' if 'sft' in args.weight else ('pretrain' if 'pretrain' in args.weight else '')
        ckp = f'{args.save_dir}/{task_dir}/{args.weight}_{args.hidden_size}.pth' if task_dir else f'{args.save_dir}/{args.weight}_{args.hidden_size}.pth'

        # 加载模型权重到模型实例中
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')

    # 如果不是本地model路径，就直接用transformers库的AutoModelForCausalLM从指定路径加载模型，注意这个不是用我们的模型架构了
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="SelfMiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（为0则使用默认（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = [] # 用于存储多轮对话的历史记录
    model, tokenizer = init_model(args) # 加载模型和分词器
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    # 初始化流式输出器（TextStreamer）
    # skip_prompt=True：流式打印时，不要把用户刚刚输入的问题再打印一遍。skip_special_tokens=True：不要打印 <eos>、<bos> 等控制字符。
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 模式0：遍历预设的 prompts 列表。
    # 模式1：进入一个无限循环，等待用户输入问题，直到用户输入特定的结束命令（如 "exit"）来退出循环。
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(2026) # or setup_seed(random.randint(0, 2048)) 
        if input_mode == 0: print(f'💬: {prompt}')
        
        # 保留最近的args.historys条历史记录
        conversation = conversation[-args.historys:] if args.historys else []
        # 将当前用户的新问题，以 OpenAI 规范的字典格式追加到对话列表中
        conversation.append({"role": "user", "content": prompt})

        '''
        一个字典，把需要分词器处理的所有要求放在字典里
        "tokenize": False：先按模板把聊天记录拼接好就行，先不要急着把它变成一堆数字
        "add_generation_prompt": True：把前面的对话拼接完之后，请在最末尾强行加上一个代表 AI 身份的标签（比如 <|im_start|>assistant\n）”
        '''
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason': templates["enable_thinking"] = True # 仅Reason模型使用
        # 如果是预训练版本，要在prompt前面加上bos；如果是微调后的，就把上面的对话模板传进去
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"], # 输入tokenid和mask
            max_new_tokens=args.max_new_tokens,  #最大输出多少新token
            do_sample=True,  #概率采样，如果是False就是贪心解码了
            streamer=streamer,#流式输出器，配合上面定义的TextStreamer，可以在生成过程中实时打印输出
            pad_token_id=tokenizer.pad_token_id, #生成时用哪个token来填充（一般是pad_token_id）
            eos_token_id=tokenizer.eos_token_id,#生成到什么token就结束
            top_p=args.top_p, temperature=args.temperature, 
            repetition_penalty=1.0 #重复惩罚，默认1.0表示不使用，>1.0会惩罚模型重复生成同样的内容
        )

        # 将输出解码
        # [0]：因为默认是在一个 batch（批次）里跑的，所以取第 0 个样本的结果
        # len(inputs["input_ids"][0])是问题的长度，所以从生成的tokenid列表里把前面问题的部分切掉，只保留模型新生成的回答部分
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        
        # 预训练分支根本就不读这个列表，这是为微调后的准备的。
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()