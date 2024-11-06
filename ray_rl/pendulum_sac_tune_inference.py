import os
import sys

import ray
from ray.rllib.algorithms.sac import SAC
import numpy as np
import warnings

sys.path.append('../')
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from ray_rl.pendulum_sac_tune import get_sac_config, inference_sac, save_metrics, TrainingConfig


def safe_restore_sac(checkpoint_path: str, config, use_tune: bool = False):
    """
    安全地从checkpoint恢复SAC模型
    """
    try:
        # 忽略警告
        warnings.filterwarnings('ignore')

        # 创建SAC配置
        if use_tune:
            sac_config = get_sac_config(config, use_tune=True)
        else:
            sac_config = get_sac_config(config, use_tune=False)

        # 构建算法实例
        algo = sac_config.build()

        # 确保checkpoint_path是字符串类型
        checkpoint_path = str(checkpoint_path)

        # 尝试恢复模型
        print(f"Attempting to restore from checkpoint: {checkpoint_path}")
        algo.restore(checkpoint_path)
        print("Successfully restored model")

        return algo

    except AttributeError as e:
        print(f"Error restoring model: {str(e)}")
        print("Trying alternative restoration method...")

        # 如果常规方法失败，尝试替代方法
        try:
            # 重新初始化Ray（如果需要）
            if not ray.is_initialized():
                ray.init()

            # 使用基础配置
            basic_config = {
                "env": "Pendulum-v1",
                "framework": "torch",
                "num_gpus": 0,
                "num_workers": 0,
            }

            algo = SAC(config=basic_config)
            algo.restore(checkpoint_path)
            return algo

        except Exception as e2:
            print(f"Alternative restoration also failed: {str(e2)}")
            raise

    except Exception as e:
        print(f"Unexpected error during model restoration: {str(e)}")
        raise


def safe_inference(checkpoint_path: str, config, use_tune: bool = False, num_episodes: int = 10, try_render: bool = False):
    """
    安全地执行推理
    """
    try:
        # 初始化Ray（如果需要）
        if not ray.is_initialized():
            ray.init()

        # 恢复模型
        algo = safe_restore_sac(checkpoint_path, config, use_tune)

        if algo is None:
            print("Failed to restore model")
            return

        # 执行推理
        performance_metrics = inference_sac(algo, num_episodes=num_episodes, try_render=try_render)

        # 保存指标
        filename = "performance_metrics_tune.txt" if use_tune else "performance_metrics_no_tune.txt"
        save_metrics(performance_metrics, filename=filename)

        return performance_metrics

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise
    finally:
        # 确保Ray被正确关闭
        if ray.is_initialized():
            ray.shutdown()


# 使用示例
if __name__ == "__main__":
    config = TrainingConfig()

    # 从tune checkpoint恢复并推理
    tune_checkpoint = "/Users/xnpeng/sumoptis/sumo-rl/ray_results/pendulum_sac_tune/SAC_Pendulum-v1_c3025_00000_0_initial_alpha=0.4881,actor_learning_rate=0.0004,critic_learning_rate=0.0004,entropy_learning_rate=0._2024-11-04_15-16-51/checkpoint_000000"
    try:
        metrics_tune = safe_inference(
            tune_checkpoint,
            config,
            use_tune=True,
            num_episodes=10,
            try_render=False
        )
    except Exception as e:
        print(f"Failed to run tune inference: {str(e)}")

    # 从no-tune checkpoint恢复并推理
    # no_tune_checkpoint = "/Users/xnpeng/sumoptis/sumo-rl/ray_results/pendulum_sac_no_tune/checkpoint_0"
    # try:
    #     metrics_no_tune = safe_inference(
    #         no_tune_checkpoint,
    #         config,
    #         use_tune=False,
    #         num_episodes=10,
    #         try_render=False
    #     )
    # except Exception as e:
    #     print(f"Failed to run no-tune inference: {str(e)}")