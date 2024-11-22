from setuptools import find_packages, setup

setup(
    name="scadl",
    version="0.1",
    description="A tool for state of the art deep learning based side-channel attacks",
    author="Karim M. Abdellatif",
    packages=find_packages(),
    install_requires=["numpy", "keras", "matplotlib", "tqdm", "h5py"],
)
