from setuptools import setup, find_packages

setup(
    name="hate-speech-detection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0,<2.0.0",
        "pandas>=1.5.0,<2.0.0", 
        "numpy>=1.21.0,<1.25.0",
        "matplotlib>=3.5.0,<3.8.0",
        "seaborn>=0.11.0,<0.13.0",
        "plotly>=5.0.0,<6.0.0",
        "scikit-learn>=1.0.0,<1.4.0",
        "tensorflow>=2.10.0,<2.15.0",
        "wordcloud>=1.9.0,<2.0.0",
        "pillow>=9.0.0,<11.0.0",
    ],
    python_requires=">=3.8,<3.12",
)