from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]  # 真实的权重
true_b = 4.2  # 真实的偏差
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))  # 生成特征
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 生成标签
labels += nd.random.normal(scale=0.01, shape=labels.shape)  # 加上噪声

print(features[0], labels[0])


def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1);  # 加分号只显示图


# 读取数据
from mxnet.gluon import data as gdata

batch_size = 10
# 将训练数据的特征和标签组合
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
