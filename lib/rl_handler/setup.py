import pathlib

from setuptools import find_packages, setup


CWD = pathlib.Path(__file__).absolute().parent


setup(
    name="epistemic_frontier_rl_handler",
    version="0.1.0",
    description="Offline RL-based frontier selector",
    long_description=(CWD / "README.md").read_text(encoding="utf-8")
    if (CWD / "README.md").exists()
    else "Offline RL-based frontier selector",
    url="https://github.com/FrancescoFabiano/deep/lib/rl_handler",
    license="GPLv3",
    author="Giovanni Briglia",
    author_email="giovanni.briglia@phd.unipi.it",
    packages=find_packages(),
    install_requires=["torch"],
    include_package_data=True,
)
