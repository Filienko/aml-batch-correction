"""Setup script for SCCL."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "sccl" / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text()
else:
    long_description = "SCCL: Single Cell Classification Library"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
else:
    requirements = [
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'anndata>=0.8.0',
        'scanpy>=1.9.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'pyyaml>=5.4.0',
    ]

# Optional dependencies
extras_require = {
    'scimilarity': ['scimilarity'],
    'scvi': ['scvi-tools>=1.0.0'],
    'all': ['scimilarity', 'scvi-tools>=1.0.0'],
    'dev': ['pytest', 'black', 'flake8', 'sphinx'],
}

setup(
    name='sccl',
    version='0.1.0',
    description='Single Cell Classification Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/Filienko/aml-batch-correction',
    packages=find_packages(exclude=['tests', 'examples', 'docs']),
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'sccl=sccl.cli.main:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
)
