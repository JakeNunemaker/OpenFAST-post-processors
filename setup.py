""""Distribution setup"""

from setuptools import setup, find_packages


setup(
    name="OpenFAST-post-processors",
    description="A collection of OpenFAST post-processors and IO functionality",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pylint",
            "flake8",
            "black",
            "isort",
            "pytest",
            "pytest-cov",
            "sphinx",
            "sphinx-rtd-theme",
        ]
    },
    test_suite="pytest",
    tests_require=["pytest", "pytest-cov"],
)
