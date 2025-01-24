from setuptools import setup, find_packages

setup(
    name="Clinical_data_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas", "numpy", "scikit-learn", "torch", "fastapi", "uvicorn", "streamlit",
        "matplotlib", "seaborn", "shap"
    ],
)
