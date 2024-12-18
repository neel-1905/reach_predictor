from setuptools import setup, find_packages

setup(
    name='reach_predictor',
    version='0.1.0',
    description='A package for predicting Instagram post reach (impressions).',
    author='Your Name',
    author_email='neelnarwadkar@gmail.com',
    url='https://github.com/neel-1905/reach_predictor',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
