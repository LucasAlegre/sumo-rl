import datetime
import json
import ntpath
import os
import re
from pathlib import Path
from typing import Dict


def ensure_dir(directory: str) -> str:
    """Ensure directory exists and return its path"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def save_json(data: Dict, filepath: str):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def write_evaluation(results: Dict, filepath: str):
    """Write evaluation results to file"""
    with open(filepath, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")


def extract_crossname_from_netfile(path):
    # 使用 ntpath.basename 来处理 Windows 路径
    filename = ntpath.basename(path)
    # 分割文件名和扩展名
    name_parts = filename.split('.')
    # 返回第一个部分（基本文件名）
    return name_parts[0]


def extract_crossname_from_evalfile(filename):
    # 使用正则表达式匹配文件名模式
    match = re.match(r'(.*?)-eval-', filename)
    if match:
        return match.group(1)
    else:
        # 如果没有匹配到预期的模式，返回None或者引发一个异常
        return None


def make_sub_dir(name) -> str:
    # file_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.getcwd()
    name_dir = os.path.join(file_dir, name)
    os.makedirs(name_dir, exist_ok=True)
    if isinstance(name_dir, bytes):
        name_dir = name_dir.decode('utf-8')  # 解码为字符串
    return name_dir


def create_file(dir_and_filename: str):
    # 规范化路径分隔符
    path = os.path.normpath(dir_and_filename)
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    # 确保文件所在的目录存在
    os.makedirs(directory, exist_ok=True)  # 创建目录及其父目录（如果不存在）
    with open(filename, 'a') as f:
        return


def write_eval_result(mean, std, filename="eval_result.txt"):
    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 将时间和变量组合成一行
    line = f"{current_time}, {mean}, {std}\n"
    create_file(filename)
    # 以写入模式打开文件并写入
    with open(filename, "a") as file:
        file.write(line)
    print(f"Data written to {filename}")


def write_predict_result(data, filename='predict_results.json', print_to_console=False):
    create_file(filename)

    if print_to_console:
        print(data)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def change_file_extension(file_name, new_extension):
    base_name, _ = os.path.splitext(file_name)
    new_file_name = base_name + '.' + new_extension
    return new_file_name


def write_loop_state(state_list, filename="predict_loop.txt"):
    filename = change_file_extension(filename, "txt")
    create_file(filename)
    with open(filename, "a") as file:
        for line in state_list:
            file.write(line)
    print(f"Data written to {filename}")
