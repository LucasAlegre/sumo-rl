import os


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


# 测试函数
def test_create_file():
    test_cases = [
        "example.txt",
        "folder/example.txt",
        "folder1/folder2/example.txt",
        "C:\\Users\\YourName\\Documents\\example.txt"  # Windows 路径示例
    ]

    for case in test_cases:
        print(f"\nTesting with filename: {case}")
        create_file_if_not_exists(case)


# 运行测试
if __name__ == "__main__":
    test_create_file()
