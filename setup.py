import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rlbrainmaturation-jiajiexiao",
    version="0.0.1",
    author="Jiajie Xiao",
    author_email="jiajiexiao@gmail.com",
    description="A rl package for brain maturation modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jiajiexiao/RL_BrainMaturation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)