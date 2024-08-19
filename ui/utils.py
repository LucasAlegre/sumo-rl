import datetime
import json
import ntpath
import os
import re
from pathlib import Path
import gradio as gr


def extract_crossname_from_netfile(path):
    # 使用 ntpath.basename 来处理 Windows 路径
    filename = ntpath.basename(path)
    # 分割文件名和扩展名
    name_parts = filename.split('.')
    # 返回第一个部分（基本文件名）
    return name_parts[0]


def create_file_if_not_exists(filename):
    # 获取文件所在的目录路径
    directory = os.path.dirname(filename)
    # 如果目录不存在，创建目录
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            return False
    # 如果文件不存在，创建文件
    if not os.path.exists(filename):
        try:
            with open(filename, 'w') as f:
                pass  # 创建一个空文件
            print(f"Created file: {filename}")
        except IOError as e:
            print(f"Error creating file {filename}: {e}")
            return False
    else:
        print(f"File already exists: {filename}")
    return True


def add_directory_if_missing(path, directory="./"):
    # 规范化路径分隔符
    path = os.path.normpath(path)
    # 分割路径
    path_parts = os.path.split(path)
    # 检查是否已经包含目录
    if path_parts[0]:
        dir_path = os.path.join(directory, path_parts[0])
    else:
        # 如果没有目录，添加指定的目录
        dir_path = directory
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            print(f"Directory created: {dir_path}")
        except OSError as e:
            print(f"Error creating directory {dir_path}: {e}")

    return dir_path


def write_eval_result(mean, std, filename="eval_result.txt"):
    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 将时间和变量组合成一行
    line = f"{current_time}, {mean}, {std}\n"

    create_file_if_not_exists(filename)
    # 以写入模式打开文件并写入
    with open(filename, "a") as file:
        file.write(line)
    print(f"Data written to {filename}")


def write_predict_result(data, filename='predict_results.json', print_to_console=False):
    create_file_if_not_exists(filename)

    if print_to_console:
        print(data)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def get_relative_path(file_path):
    # 获取当前工作目录
    current_dir = Path.cwd()

    # 将文件路径转换为Path对象
    file_path = Path(file_path)

    # 尝试获取相对路径
    try:
        relative_path = file_path.relative_to(current_dir)
    except ValueError:
        # 如果无法获取相对路径（例如，文件在不同的驱动器上），则返回原始路径
        relative_path = file_path

    # 分离文件夹和文件名
    folder = str(relative_path.parent)
    filename = relative_path.name

    # 如果文件夹是当前目录，则用 './' 表示
    if folder == '.':
        folder = './'
    elif folder == '..':
        folder = '../'
    else:
        folder = f'./{folder}'

    return folder, filename


def extract_crossname_from_evalfile(filename):
    # 使用正则表达式匹配文件名模式
    match = re.match(r'(.*?)-eval-', filename)
    if match:
        return match.group(1)
    else:
        # 如果没有匹配到预期的模式，返回None或者引发一个异常
        return None


def get_gradio_file_info(file: gr.File):
    if file is None:
        return None, None

    # 获取原始文件名
    filename = os.path.basename(file.name)

    conn_ep = r'_conn(\d+)_ep(\d+)'

    # 推断预期的文件夹
    if 'eval' in filename:
        inferred_folder = './evals'
    elif 'predict' in filename:
        inferred_folder = './predicts'
    elif re.search(conn_ep, filename):
        inferred_folder = './outs'
    else:
        inferred_folder = './'  # 默认为当前目录

    return inferred_folder, filename
