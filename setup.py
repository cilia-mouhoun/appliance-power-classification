from setuptools import setup, find_packages

setup(
    name='appliance-power-classification',
    version='0.1.0',
    description='Time-series classification for appliance power consumption data',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
        'xgboost>=1.5.0',
        'joblib>=1.1.0',
    ],
    extras_require={
        'dev': [
            'jupyter>=1.0.0',
            'ipython>=7.0.0',
            'pytest>=6.2.0',
        ],
    },
)
