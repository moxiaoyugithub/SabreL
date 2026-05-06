import os
import re
import shutil

def read_all_file(directory):
    # 读取目录下所有文件和文件夹
    files_and_folders = os.listdir(directory)
    # 过滤出文件（排除文件夹）v
    file_names = [f for f in files_and_folders if os.path.isfile(os.path.join(directory, f))]
    return file_names

# 删除并重建文件夹
def delete_and_recreate_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # 删除整个文件夹及其所有内容
        os.makedirs(folder_path, exist_ok=True)  # 重新创建空文件夹
    except Exception as e:
        print(f'Failed to delete and recreate folder {folder_path}. Reason: {e}')

def find_latest_checkpoint(checkpoint_dir):
    """
    搜索目录下 j 最大的检查点文件
    :param checkpoint_dir: 检查点存放目录
    :return: (latest_checkpoint_path, max_j) 如果没找到则返回 (None, 0)
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0

    files = os.listdir(checkpoint_dir)
    # 正则匹配格式：sabre_agent_update_{数字}.pth
    pattern = re.compile(r"sabre_agent_update_(\d+)\.pth")
    
    max_j = -1
    latest_file = None

    for f in files:
        match = pattern.match(f)
        if match:
            j_val = int(match.group(1))
            if j_val > max_j:
                max_j = j_val
                latest_file = f

    if latest_file:
        full_path = os.path.join(checkpoint_dir, latest_file)
        return full_path, max_j
    else:
        return None, 0