from setuptools import setup

setup(
    name='longbart',
    version='0.1',
    description='Long version of the BART model',
    url='https://github.com/patil-suraj/longbart',
    author='Suraj Patil',
    author_email='surajp815@gmail.com',
    packages=['longbart'],
    keywords="NLP deep learning transformer pytorch bart",
    install_requires=[
        'transformers == 4.30.0'
    ],
    python_requires=">=3.6.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False
)