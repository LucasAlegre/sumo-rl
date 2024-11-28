import pandas as pd


def convert(file_path):
    # 读取文件时，使用正则分隔符（多个空格），并确保日期时间字段不被分开
    df = pd.read_csv(file_path, sep=r'\s{2,}', header=None, engine='python')

    # 打印文件的前几行，以确认列的分隔情况
    print(df.head())

    # 重命名列以便更方便处理
    df.columns = ['行号', '路口号', '日期时间', '车道号', '车牌', '车身颜色', '车型', '车主类型', '无用字段1', '无用字段2']

    # 定义路口号和车道号的对应关系
    intersection_map = {
        'NS': ['213', '224', '225', '226'],
        'NE': ['211', '212'],
        'EW': ['232', '243', '244'],
        'ES': ['231'],
        'SN': ['253', '264', '265', '266'],
        'SW': ['251', '252'],
        'WE': ['273', '284', '285', '286'],
        'WN': ['271', '272']
    }

    # 定义一个函数来映射路口号和车道号
    def map_intersection(row):
        key = str(row['路口号']) + str(row['车道号'])
        for direction, codes in intersection_map.items():
            if key in codes:
                return direction
        return '未知'  # 如果没有匹配的，返回'未知'

    # 提取所需字段
    df_output = df[['日期时间', '路口号', '车道号', '车牌']].copy()

    # 创建新列来存储方向
    df_output['路口号+车道号'] = df_output.apply(map_intersection, axis=1)

    # 选择所需的列，格式化输出
    df_output = df_output[['日期时间', '路口号+车道号', '车牌']]

    # 打印文件的前几行，以确认列的分隔情况
    print(df_output.head())

    # 输出为新的文本文件
    df_output.to_csv('./zszx/data/output_data.txt', sep='\t', header=False, index=False)

    print("数据已成功转换并保存到 output_data.txt")


def minute_flow(file_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import matplotlib.dates as mdates

    # 设置支持中文字体，避免乱码
    rcParams['font.sans-serif'] = ['KaiTi']  # 使用SimHei字体，支持中文
    rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    # 读取数据文件
    df_output = pd.read_csv(file_path, sep='\t', header=None, names=['日期时间', '路口号+车道号', '车牌'])

    # 由于'路口号+车道号'和'日期时间'可能是被误混合的，我们需要将它们分开。
    # 使用正则表达式从'日期时间'列中提取出日期时间部分，并移除方向信息
    df_output['日期时间'] = df_output['日期时间'].str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})')[0]
    df_output['路口号+车道号'] = df_output['路口号+车道号'].str.extract(r'([A-Za-z]+)')[0]  # 提取方向信息

    print(df_output.head())
    # 将日期时间列转换为日期时间格式
    df_output['日期时间'] = pd.to_datetime(df_output['日期时间'])
    print(df_output.head())

    # 生成分钟列
    df_output['分钟'] = df_output['日期时间'].dt.floor('T')  # 按分钟进行聚合

    # 按方向和分钟统计车流量
    traffic_flow_minute = df_output.groupby([df_output['分钟'], '路口号+车道号']).size().reset_index(name='车流量')
    traffic_flow_minute.to_csv('./zszx/flow/traffic_flow_minute.csv', index=False)

    # 获取所有流向（方向）
    directions = traffic_flow_minute['路口号+车道号'].unique()

    # 绘制每个方向的时间-车流量曲线
    plt.figure(figsize=(24, 8))

    for direction in directions:
        # 获取该方向的数据
        direction_data = traffic_flow_minute[traffic_flow_minute['路口号+车道号'] == direction]

        # 绘制该方向的时间-车流量曲线
        plt.plot(direction_data['分钟'], direction_data['车流量'], label=direction)

    # 设置图形的标题和标签
    plt.title('各方向车流量分钟序列')
    plt.xlabel('时间')
    plt.ylabel('车流量')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.legend(title='方向')

    # 格式化X轴的时间显示，设置为分钟
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H-%M'))  # 只显示小时和分钟
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())  # 设置每小时为一个刻度

    # 显示图形
    plt.tight_layout()
    plt.savefig('./zszx/flow/traffic_flow_minute.png')
    plt.show()


def hourly_flow(file_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import matplotlib.dates as mdates

    # 设置支持中文字体，避免乱码
    rcParams['font.sans-serif'] = ['KaiTi']  # 使用SimHei字体，支持中文
    rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    # 读取数据文件
    df_output = pd.read_csv(file_path, sep='\t', header=None, names=['日期时间', '路口号+车道号', '车牌'])

    # 由于'路口号+车道号'和'日期时间'可能是被误混合的，我们需要将它们分开。
    # 使用正则表达式从'日期时间'列中提取出日期时间部分，并移除方向信息
    df_output['日期时间'] = df_output['日期时间'].str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})')[0]
    df_output['路口号+车道号'] = df_output['路口号+车道号'].str.extract(r'([A-Za-z]+)')[0]  # 提取方向信息

    # 将日期时间列转换为日期时间格式
    df_output['日期时间'] = pd.to_datetime(df_output['日期时间'])

    # 生成小时列
    df_output['小时'] = df_output['日期时间'].dt.floor('H')  # 按小时进行聚合

    # 按方向和小时统计车流量
    traffic_flow_hourly = df_output.groupby([df_output['小时'], '路口号+车道号']).size().reset_index(name='车流量')
    traffic_flow_hourly.to_csv('./zszx/flow/traffic_flow_hourly.csv', index=False)

    # 获取所有流向（方向）
    directions = traffic_flow_hourly['路口号+车道号'].unique()

    # 绘制每个方向的时间-车流量曲线
    plt.figure(figsize=(24, 8))

    for direction in directions:
        # 获取该方向的数据
        direction_data = traffic_flow_hourly[traffic_flow_hourly['路口号+车道号'] == direction]

        # 绘制该方向的时间-车流量曲线
        plt.plot(direction_data['小时'], direction_data['车流量'], label=direction)

    # 设置图形的标题和标签
    plt.title('各方向车流量小时序列')
    plt.xlabel('时间')
    plt.ylabel('车流量')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.legend(title='方向')

    # 格式化X轴的时间显示，设置为小时
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H-%M'))  # 只显示小时和分钟
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())  # 设置每小时为一个刻度

    # 显示图形
    plt.tight_layout()
    plt.savefig('./zszx/flow/traffic_flow_hourly.png')
    plt.show()


def daily_flow(file_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # 设置支持中文字体，避免乱码
    rcParams['font.sans-serif'] = ['KaiTi']  # 使用SimHei字体，支持中文
    rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    # 读取数据文件
    df_output = pd.read_csv(file_path, sep='\t', header=None, names=['日期时间', '路口号+车道号', '车牌'])

    # 使用正则表达式从'日期时间'列中提取出日期时间部分，并移除方向信息
    df_output['日期时间'] = df_output['日期时间'].str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})')[0]
    df_output['路口号+车道号'] = df_output['路口号+车道号'].str.extract(r'([A-Za-z]+)')[0]  # 提取方向信息

    # 将日期时间列转换为日期时间格式
    df_output['日期时间'] = pd.to_datetime(df_output['日期时间'])

    # 提取日期和小时部分
    df_output['日期'] = df_output['日期时间'].dt.date

    # 按日和方向统计车流量
    traffic_flow_daily = df_output.groupby([df_output['日期'], '路口号+车道号']).size().reset_index(name='日流量')
    traffic_flow_daily['小时流量'] = (traffic_flow_daily['日流量'] / 24).astype(int)
    traffic_flow_daily['小时流量*4/3'] = (traffic_flow_daily['日流量'] / 18).astype(int)
    traffic_flow_daily['小时流量*2'] = (traffic_flow_daily['日流量'] / 12).astype(int)

    # 打印统计结果
    print(traffic_flow_daily)

    # 将统计结果保存为CSV文件
    traffic_flow_daily.to_csv('./zszx/flow/traffic_flow_daily.csv', index=False)

    # 绘制流量曲线
    directions = traffic_flow_daily['路口号+车道号'].unique()  # 获取所有方向
    plt.figure(figsize=(12, 8))

    for direction in directions:
        # 获取当前方向的数据
        direction_data = traffic_flow_daily[traffic_flow_daily['路口号+车道号'] == direction]

        # 绘制当前方向的车流量曲线
        plt.plot(direction_data['日期'], direction_data['日流量'], marker='o', label=direction)

    # 设置图形的标题和标签
    plt.title('各方向车流量小时序列')  # 设置中文标题
    plt.xlabel('日期')  # 设置中文X轴标签
    plt.ylabel('车流量')  # 设置中文Y轴标签
    plt.legend(title='方向')  # 设置图例

    # 显示图形
    plt.xticks(rotation=45)  # 让日期刻度倾斜，避免重叠
    plt.tight_layout()
    plt.savefig('./zszx/flow/traffic_flow_daily.png')
    plt.show()


if __name__ == '__main__':
    input_file = './zszx/data/input_data.txt'
    # convert()
    output_file = './zszx/data/output_data.txt'
    # minute_flow(output_file)
    # hourly_flow(output_file)
    daily_flow(output_file)
