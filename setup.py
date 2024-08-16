import pathlib

import pkg_resources
from setuptools import setup, find_packages

PKG_NAME = "uvd"
VERSION = "0.0.1"


def _read_file(fname):
    with pathlib.Path(fname).open() as fp:
        return fp.read()


def _read_install_requires():
    with pathlib.Path("requirements.txt").open() as fp:
        main = [
            str(requirement) for requirement in pkg_resources.parse_requirements(fp)
        ]
    return main


setup(
    name=PKG_NAME,
    version=VERSION,
    author=f"{PKG_NAME} Developers",
    # url='http://github.com/',
    description="research project",
    long_description=_read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords=["Deep Learning", "Reinforcement Learning"],
    license="MIT License",
    packages=find_packages(include=f"{PKG_NAME}.*"),
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            # 'cmd_tool=mylib.subpkg.module:main',
        ]
    },
    install_requires=_read_install_requires(),
    python_requires=">=3.9.*",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
    ],
)
