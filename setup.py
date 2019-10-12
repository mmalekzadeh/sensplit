import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sensplit",
    version="0.0.4",
    author="Mohammad Malekzadeh",
    author_email="moh.malekzadeh@gmail.com",
    description="Splits a dataset (in Pandas dataframe format) to train/test sets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sensplit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)