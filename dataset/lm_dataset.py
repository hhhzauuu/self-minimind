from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer #分词器
        self.max_length = max_length 
        self.samples = load_dataset('json', data_files=data_path, split='train')#加载数据集到self.samples，data_path是json文件的路径，split='train'表示加载训练集

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index] #根据索引获取数据

        # 提取到文本后，进行分词，转为tokenid
        # add_special_tokens=False表示不添加特殊token(我们自己添加)
        # max_length=self.max_length - 2表示文本的最大长度，减去2是因为我们要添加bos_token和eos_token
        # .input_ids表示只获取分词后的tokenid列表,因为我们要手搓
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        # 在分词列表的头部加上 bos_token_id，尾部加上 eos_token_id
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 补齐到max_length，使用pad_token_id进行填充
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        # 把list转为torch.tensor，并且设置数据类型为long
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 因为自回归模型的输入数据就是标签
        # 我们直接复制一份，模型内部会进行shift对齐
        labels = input_ids.clone()
        # 贴标签：将padding部分的标签设置为-100，模型在计算损失时会忽略这些位置
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
    
'''
注入系统提示词：目的是在对话数据前，以一定概率随机添加system角色设定。
'''
def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 即使对话中没有system设定，也不能一股脑直接把system加进去
    # 一方面能够模拟更真实的对话场景（减少依赖），另一方面能够防止过拟合。
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations
'''
处理空的思考标签
'''
def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train') #加载jsonl格式的数据集，jsonl格式是每行一个json对象，split='train'表示加载训练集
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids # 找到<|im_start|>assistant的tokenid，是模型生成的起点
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)
    
    ''' 把对话列表转为模型能看懂的字符串'''
    def create_chat_prompt(self, conversations):
        messages = conversations.copy()
        
        # 如果对话的第一条是system角色，并且里面有functions字段，就把functions字段里的工具信息也传给tokenizer
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    '''SFT的核心：把用户提问以及系统提示词都屏蔽掉'''
    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids) #先初始化为-100
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id: #找到了模型回答的开始
                start = i + len(self.bos_id)# 模型回答的开始位置
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 把模型回答的部分的标签，改为真实值
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                # i进行下一句的判断
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations']) #注入system，然后就是完整的对话
        prompt = self.create_chat_prompt(conversations)# 把对话的列表格式转为大模型能理解的字符串
        prompt = post_processing_chat(prompt) #处理空的思考标签
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]#把字符串转为tokenid，并且截断到max_length
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids) #生成标签
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
