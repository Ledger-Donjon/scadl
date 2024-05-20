from setuptools import setup, find_packages

setup(
    name = "scadl",
    version = "0.1",
    description  = "A tool for state of the artr DL attacks",
    author = "Karim M. Abdellatif",
    packages = ["scadl"],
    install_requires = [
        "numpy",
        "keras",
        "matplotlib"
    ]
)