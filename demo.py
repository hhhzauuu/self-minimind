from modelscope.hub.file_download import dataset_file_download

file_path = dataset_file_download(
    dataset_id='gongjy/minimind_dataset',
    file_path='sft_mini_512.jsonl',
    local_dir='./dataset'
)

print(f"✅ 文件已下载")