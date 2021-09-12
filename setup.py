from setuptools import setup, find_packages

REQUIRED = ['gym', 'numpy', 'pandas']

extras = {
    "rllib": ['ray[rllib]'],
    "pettingzoo": ["pettingzoo"],
}

extras["all"] = extras["rllib"]+extras["pettingzoo"]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='sumo-rl',
    version='1.0',
    packages=['sumo_rl'],
    install_requires=REQUIRED,
    extras_require=extras,
    author='LucasAlegre',
    author_email='lucasnale@gmail.com',
    url='https://github.com/LucasAlegre/sumo-rl',
    download_url='https://github.com/LucasAlegre/sumo-rl/archive/v1.0.tar.gz',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    description='RL environment wrappers and learning code traffic signal Control in SUMO.'
)
