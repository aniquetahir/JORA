from setuptools import setup, find_packages
import re

_deps = [
    "torch>=2.1.0",
    "jax>=0.4.25",
    "optax>=0.2.1",
    "tqdm>=4.62.3",
    "transformers>=4.37.0",
    "fire>=0.5.0",
    "numpy>=1.21.2",
    "einops>=0.6.1"
]
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


setup(
    name='jora',
    version='0.1.0',
    author='Anique Tahir',
    author_email='research@anique.org',
    description='JORA: JAX Tensor-Parallel LoRA Library for Fine-Tuning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/aniquetahir/jora',
    package_dir={"": "jora"},
    packages=find_packages("jora"),
    license='Creative Commons NonCommercial License',
    classifiers=[
        'License :: Other/Proprietary License',
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        # list of packages your project depends on
        deps['torch'],
        deps['jax'],
        deps['optax'],
        deps['tqdm'],
        deps['transformers'],
        deps['fire'],
        deps['numpy'],
        deps['einops']
    ],
)