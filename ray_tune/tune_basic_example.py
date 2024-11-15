"""This example demonstrates basic Ray Tune random search and grid search."""
import time

import ray
from ray import train, tune

# 核心评估函数
def evaluation_fn(step, width, height):
    time.sleep(0.1)
    return (0.1 + width * step / 100) ** (-1) + height * 0.1

# 目标函数：从配置中获取超参数，执行迭代训练过程，计算评估值，报告中间结果
def easy_objective(config):
    # Hyperparameters
    width, height = config["width"], config["height"]

    for step in range(config["steps"]):
        # Iterative training function - can be any arbitrary training procedure
        intermediate_score = evaluation_fn(step, width, height)
        # Feed the score back back to Tune.
        train.report({"iterations": step, "mean_loss": intermediate_score})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    args, _ = parser.parse_known_args()

    ray.init(configure_logging=False)

    # This will do a grid search over the `activation` parameter. This means
    # that each of the two values (`relu` and `tanh`) will be sampled once
    # for each sample (`num_samples`). We end up with 2 * 50 = 100 samples.
    # The `width` and `height` parameters are sampled randomly.
    # `steps` is a constant parameter.

    tuner = tune.Tuner(
        easy_objective,
        tune_config=tune.TuneConfig(
            metric="mean_loss",
            mode="min",
            num_samples=5 if args.smoke_test else 50,
        ),
        param_space={
            "steps": 5 if args.smoke_test else 100,
            "width": tune.uniform(0, 20),
            "height": tune.uniform(-100, 100),
            "activation": tune.grid_search(["relu", "tanh"]),
        },
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)

"""
width: 在 0-20 之间均匀随机采样
height: 在 -100 到 100 之间均匀随机采样
activation: 网格搜索 ["relu", "tanh"]
steps: 固定值(正常运行是 100,测试模式是 5)

smoke-test
Current best trial: 33e91_00000 with mean_loss=-5.980934559524437 and params={'steps': 5, 'width': 6.766931800086016, 'height': -86.78699098998868, 'activation': 'relu'}
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                   status          width      height   activation         loss     iter     total time (s)     iterations │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ easy_objective_33e91_00000   TERMINATED    6.76693   -86.787     relu           -5.98093        5           0.521663              4 │
│ easy_objective_33e91_00001   TERMINATED   13.3517     59.5754    tanh            7.53466        5           0.506344              4 │
│ easy_objective_33e91_00002   TERMINATED    5.82504   -85.5401    relu           -5.55102        5           0.517627              4 │
│ easy_objective_33e91_00003   TERMINATED   15.6171     58.3202    tanh            7.21193        5           0.519991              4 │
│ easy_objective_33e91_00004   TERMINATED   13.7775    -36.8709    relu           -2.15123        5           0.514528              4 │
│ easy_objective_33e91_00005   TERMINATED   11.7008     98.8172    tanh           11.6422         5           0.509735              4 │
│ easy_objective_33e91_00006   TERMINATED   13.4066      7.40458   relu            2.31213        5           0.520584              4 │
│ easy_objective_33e91_00007   TERMINATED    8.08537    57.0109    tanh            8.06284        5           0.520432              4 │
│ easy_objective_33e91_00008   TERMINATED    1.84714   -15.551     relu            4.19581        5           0.52527               4 │
│ easy_objective_33e91_00009   TERMINATED    8.73038    89.7982    tanh           11.2059         5           0.519572              4 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Best hyperparameters found were:  {'steps': 5, 'width': 6.766931800086016, 'height': -86.78699098998868, 'activation': 'relu'}

"""