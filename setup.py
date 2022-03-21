from setuptools import setup, find_packages

REQUIRED = ['gym', 'numpy', 'pandas', 'pillow', 'pettingzoo']

extras = {
    "rendering": ["pyvirtualdisplay"]
}
extras["all"] = extras["rendering"]

setup(
    name='sumo-rl',
    version='1.2',
    packages=['sumo_rl'],
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
