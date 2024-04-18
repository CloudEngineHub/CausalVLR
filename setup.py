from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
 
    name="hcpcvlr",
    version="0.0.2",
    description="CausalVLR: A Toolbox and Benchmark for Visual-Linguistic Causal Reasoning (视觉-语言因果推理开源框架)",
    url="https://github.com/HCPLab-SYSU/CausalVLR",  
    author=[
            "hcp_lab",
            "liuyang",
            "chenweixing"
    ],
    author_email=[
        "hcp.sysu@gmail.com",
        "liuy856@mail.sysu.edu.cn",
        "chen867820261@gmail.com"
    ], 
    classifiers=[
     
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
   
    keywords="causal",
    # packages=['hcpcvlr'],
    # package_dir={"": "src"},  # Optional 手动指定包目录
    
    packages=find_packages(),
   
    python_requires=">=3.7, <4",  # python 版本要求
 
    install_requires=[
        "torch>=1.6.0",
        "pyyaml",
        "pandas",
        "einops",
        "scipy",
        "matplotlib",
        "dominate",
        "visdom"
        ],
)