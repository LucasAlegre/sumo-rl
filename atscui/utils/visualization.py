import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


class Visualizer:
    @staticmethod
    def plot_training_process(csv_path: str, save_path: str = None):
        """Plot training metrics from CSV file"""
        df = pd.read_csv(csv_path)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot rewards
        ax1.plot(df['step'], df['reward'], label='Reward')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.legend()

        # Plot loss
        if 'loss' in df.columns:
            ax2.plot(df['step'], df['loss'], label='Loss')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss')
            ax2.legend()

        if save_path:
            plt.savefig(save_path)
        return fig

    @staticmethod
    def plot_evaluation_results(eval_path: str, save_path: str = None):
        """Plot evaluation results"""
        with open(eval_path, 'r') as f:
            lines = f.readlines()

        metrics = {}
        for line in lines:
            key, value = line.strip().split(':')
            metrics[key.strip()] = float(value.strip())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(metrics.keys(), metrics.values())
        ax.set_ylabel('Value')
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path)
        return fig