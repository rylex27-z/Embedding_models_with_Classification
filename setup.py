from setuptools import setup, find_packages

setup(
    name="imdb_benchmark",
    version="0.1.0",
    description="Reproducible benchmarking scaffold for IMDB sentiment classification",
    author="rylex27-z",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "gensim>=4.3.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
)
