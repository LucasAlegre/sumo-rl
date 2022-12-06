from setuptools import setup, find_packages

REQUIRED = ['gymnasium>=0.26', 'pettingzoo>=1.22.2', 'numpy', 'pandas', 'pillow', 'sumolib>=1.14.0', 'traci>=1.14.0']

extras = {
    "rendering": ["pyvirtualdisplay"]
}
extras["all"] = extras["rendering"]

setup(
    name='sumo-rl',
    version='1.4.0',
    packages=[package for package in find_packages() if package.startswith("sumo_rl")] + ["nets"],
    install_requires=REQUIRED,
    extras_require=extras,
    python_requires=">=3.7",
    author='LucasAlegre',
    author_email='lucasnale@gmail.com',
    url='https://github.com/LucasAlegre/sumo-rl',
    download_url='https://github.com/LucasAlegre/sumo-rl/archive/v1.2.tar.gz',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    description='RL environments and learning code for traffic signal control in SUMO.'
)
