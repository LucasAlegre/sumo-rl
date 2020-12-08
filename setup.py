from setuptools import setup, find_packages

REQUIRED = ['gym', 'numpy', 'pandas', 'ray[rllib]']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='sumo-rl',
    version='1.0',
    packages=['sumo_rl',],
    install_requires=REQUIRED,
    author='LucasAlegre',
    author_email='lucasnale@gmail.com',
    url='https://github.com/LucasAlegre/sumo-rl',
    download_url='https://github.com/LucasAlegre/sumo-rl/archive/v1.0.tar.gz',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    description='Environments inheriting OpenAI Gym Env and RL algorithms for Traffic Signal Control on SUMO.'
)
