from setuptools import setup, find_packages

REQUIRED = ['tensorflow', 'gym', 'numpy', 'pandas', 'matplotlib', 'ray[rllib]']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='sumo-rl',
    version='0.4',
    packages=['sumo_rl',],
    install_requires=REQUIRED,
    author='LucasAlegre',
    author_email='lucasnale@gmail.com',
    long_description=long_description,
    url='https://github.com/LucasAlegre/sumo-rl',
    license="MIT",
    description='Environments inheriting OpenAI Gym Env and RL algorithms to control Traffic Signal controllers on SUMO.'
)
