from setuptools import setup, find_packages

setup(
    name="textclassifier",
    version="1.0.0",
    description="LSTM text classifier for sports match articles (PREVIEW vs REPORT)",
    packages=find_packages(),
    package_data={
        "textClassifier": [
            "saved_models/*.pt",
            "data/*.csv",
        ],
    },
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "tqdm>=4.66",
    ],
    entry_points={
        "console_scripts": [
            "textclassifier=textClassifier.cli:main",
        ],
    },
    python_requires=">=3.10",
)
