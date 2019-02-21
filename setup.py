from setuptools import setup

REQUIRED = ['gym', 'numpy', 'pandas', 'matplotlib', 'ray[rllib]']

setup(
    name='sumo-rl',
    version='0.1dev',
    packages=['agents', 'environment', 'experiments', 'exploration'],
    install_requires=REQUIRED,
    author='LucasAlegre',
    author_email='lucasnale@gmail.com',
    description='Environments inheriting OpenAI Gym Env and RL algorithms to control Traffic Signal controllers on SUMO.'
)
