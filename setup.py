from setuptools import setup, find_packages

REQUIRED = ['gym==0.24.0', 'numpy', 'pandas', 'pillow', 'pettingzoo==1.18.1', 'sumolib', 'traci']

extras = {
    "rendering": ["pyvirtualdisplay"]
}
extras["all"] = extras["rendering"]

setup(
    name='sumo-rl',
    version='1.3.0',
    packages=[package for package in find_packages() if package.startswith("sumo_rl")] + ["nets"],
    install_requires=REQUIRED,
    extras_require=extras,
    author='LucasAlegre',
    author_email='lucasnale@gmail.com',
    url='https://github.com/LucasAlegre/sumo-rl',
    download_url='https://github.com/LucasAlegre/sumo-rl/archive/v1.2.tar.gz',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    description='RL environments and learning code for traffic signal control in SUMO.'
)
