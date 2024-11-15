import numpy as np
import matplotlib.pyplot as plt


def demonstrate_uniform_sampling(min_val=0, max_val=20, n_samples=1000):
    """
    演示均匀分布采样的过程

    Args:
        min_val: 最小值
        max_val: 最大值
        n_samples: 采样数量
    """
    # 使用 numpy 的 uniform 函数进行采样
    samples = np.random.uniform(min_val, max_val, n_samples)

    # 创建直方图
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True)
    plt.axhline(y=1 / (max_val - min_val), color='r', linestyle='--',
                label=f'理论密度: {1 / (max_val - min_val):.3f}')

    plt.title(f'均匀分布采样演示 [{min_val}, {max_val}]')
    plt.xlabel('取值')
    plt.ylabel('密度')
    plt.legend()

    # 计算统计信息
    stats = {
        '平均值': np.mean(samples),
        '理论平均值': (max_val + min_val) / 2,
        '标准差': np.std(samples),
        '理论标准差': np.sqrt(((max_val - min_val) ** 2) / 12),
        '最小值': np.min(samples),
        '最大值': np.max(samples)
    }

    return samples, stats


# 进行采样并显示结果
samples, stats = demonstrate_uniform_sampling()

# 打印统计信息
for key, value in stats.items():
    print(f"{key}: {value:.3f}")
