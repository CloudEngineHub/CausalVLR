"""
CausalVLR: Causal Visual-Language Reasoning Toolbox
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os

# Read the long description from README
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        # Try different encodings
        for encoding in ['utf-8', 'utf-16-le', 'utf-16', 'latin-1']:
            try:
                with open(readme_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        # If all encodings fail, return a fallback description
        return 'CausalVLR: A unified toolbox for causal visual-language reasoning tasks'
    return 'CausalVLR: A unified toolbox for causal visual-language reasoning tasks'

# Read version from __init__.py
def read_version():
    init_path = os.path.join(os.path.dirname(__file__), 'causalvlr', '__init__.py')
    with open(init_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '0.2.0'

setup(
    name='causalvlr',
    version=read_version(),
    description='A unified toolbox for causal visual-language reasoning tasks',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    author='CausalVLR Team',
    author_email='',
    url='https://github.com/yourusername/CausalVLR',  # Update with actual repo URL
    license='MIT',
    
    # Package configuration
    packages=find_packages(include=['causalvlr', 'causalvlr.*']),
    include_package_data=True,
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Core dependencies
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'transformers>=4.30.0',
        'tokenizers>=0.13.0',
        'sentencepiece>=0.1.99',
        'huggingface-hub>=0.16.0',
        'safetensors>=0.3.0',
        'scipy>=1.10.0',
        'pandas>=2.0.0',
        'h5py>=3.8.0',
        'matplotlib>=3.7.0',
        'tqdm>=4.65.0',
        'pyyaml>=6.0',
        'pillow>=10.0.0',
        'opencv-python>=4.8.0',
        'einops>=0.6.0',
        'tabulate>=0.9.0',
        'dominate>=2.8.0',
        'visdom>=0.2.4',
    ],
    
    # Optional dependencies for specific features
    extras_require={
        'dev': [
            'pytest>=7.3.0',
            'pytest-cov>=4.1.0',
            'black>=23.3.0',
            'flake8>=6.0.0',
            'mypy>=1.3.0',
            'sphinx>=6.2.0',
        ],
        'all': [
            'torch-geometric>=2.3.0',
            'accelerate>=0.20.0',
            'peft>=0.4.0',
            'timm>=0.9.0',
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    
    # Keywords for PyPI search
    keywords=[
        'deep learning',
        'machine learning',
        'computer vision',
        'natural language processing',
        'medical report generation',
        'video question answering',
        'causal reasoning',
        'visual-language models',
        'multimodal learning',
    ],
    
    # Project URLs
    project_urls={
        'Documentation': 'https://github.com/HCPLab-SYSU/CausalVLR/docs',
        'Source': 'https://github.com/HCPLab-SYSU/CausalVLR',
        'Bug Reports': 'https://github.com/HCPLab-SYSU/CausalVLR/issues',
    },
)
