import json
import ast


def convert_to_json(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r') as file:
        content = file.read()

    # 提取document_content部分
    start = content.find('<document_content>') + len('<document_content>')
    end = content.find('</document_content>')
    data_str = content[start:end].strip()

    # 将字符串分割成行
    lines = data_str.split('\n')

    # 解析每一行并存储在列表中
    data = []
    for line in lines:
        try:
            # 使用ast.literal_eval安全地评估字符串为Python对象
            parsed_line = ast.literal_eval(line)
            # 如果解析成功，添加到数据列表
            if isinstance(parsed_line, list) and len(parsed_line) > 0:
                data.extend(parsed_line)
        except (SyntaxError, ValueError):
            print(f"Skipping invalid line: {line}")

    # 将数据写入JSON文件
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=2)

    print(f"Data has been successfully converted and written to {output_file}")


# 使用函数
input_file = 'info.txt'  # 输入文件名
output_file = 'output.json'  # 输出文件名

convert_to_json(input_file, output_file)
