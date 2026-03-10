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