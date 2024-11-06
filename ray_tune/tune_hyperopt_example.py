import time

from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

# 评估函数：根据输入计算输出值
def evaluate(step, width, height):
    time.sleep(0.1)
    return (0.1 + width * step / 100) ** (-1) + height * 0.1

# 进行超参数优化的目标函数.
# 在函数内部，根据 config["steps"] 的值，循环进行多次评估。
# 在每次循环中，调用 evaluate() 来计算得分，并通过 train.report() 报告当前迭代步数和对应的损失（即 mean_loss）。
def objective(config):
    for step in range(config["steps"]):
        score = evaluate(step, config["width"], config["height"])
        train.report({"iterations": step, "mean_loss": score})


# initial_params 是一个包含初始超参数配置的列表，这些配置将在搜索开始时优先被评估。
# HyperOptSearch 是基于 Hyperopt 的优化算法，它将尝试在给定的搜索空间内找到最优的超参数配置。通过 points_to_evaluate=initial_params，指定初始评估的点。
# ConcurrencyLimiter 用来限制并行执行的最大数量，确保不会同时进行超过 4 个评估。
initial_params = [
    {"width": 1, "height": 2, "activation": "relu"},
    {"width": 4, "height": 2, "activation": "tanh"},
]
algo = HyperOptSearch(points_to_evaluate=initial_params)  # 基于 Hyperopt 库的搜索算法，用于超参数优化
algo = ConcurrencyLimiter(algo, max_concurrent=4)  # 限制并行评估的数量

num_samples = 1000

# 定义了超参数的搜索空间
search_config = {
    "steps": 100,
    "width": tune.uniform(0, 20),
    "height": tune.uniform(-100, 100),
    "activation": tune.choice(["relu", "tanh"])
}

# 创建了一个 Tuner 对象来进行超参数调优：
# objective 是待优化的目标函数，即我们定义的 objective()。
# tune_config 配置了调优过程的参数：
# metric="mean_loss"：调优的目标是最小化 mean_loss（即 objective() 中返回的得分）。
# mode="min"：目标是最小化该指标，因此设置为 "min"。
# search_alg=algo：指定使用的搜索算法是 HyperOptSearch，并通过 ConcurrencyLimiter 限制并发度。
# num_samples=num_samples：定义了评估的总次数，设为 1000 次。
# param_space=search_config：指定了超参数搜索空间。
# tuner.fit() 启动了调优过程，Ray Tune 会根据配置自动执行超参数搜索并输出结果。
tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space=search_config,
)
results = tuner.fit()

"""
超参数优化例子--运行正常。

程序使用了 Ray Tune 进行超参数优化，具体使用了 Hyperopt 算法（通过 HyperOptSearch）进行搜索，并结合 并发限制器（ConcurrencyLimiter） 来控制并行评估的数量。

关键点总结：
Ray Tune 被用来进行超参数优化。
objective 函数用于定义评估过程，并在每次评估时通过 train.report() 报告指标（mean_loss）。
HyperOptSearch 算法用于在指定的超参数空间内进行搜索，ConcurrencyLimiter 用于限制并发评估的数量。
超参数空间包括 width、height 和 activation，其中 width 和 height 是连续值，而 activation 是类别值。

"""

