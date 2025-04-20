from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "covcomp",                        # name of the generated module
        ["src/CovarianceComputation.cpp"],    # your C++ source
        define_macros=[('VERSION_INFO', "0.1.0")],
        language="c++",
    ),
]

setup(
    name="covcomp",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="Covariance computation with Ledoit-Wolf shrinkage",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)