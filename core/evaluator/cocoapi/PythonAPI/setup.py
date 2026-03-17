import os
import platform
from setuptools import setup, Extension
import numpy as np

if platform.system() == 'Windows':
    # Options for syntax(MSVC): Remove all problematic usage options (-Wno-CPP, etc...)
    extra_compile_args = []
else:
    # Options for Linux or Mac
    extra_compile_args = ['-Wno-cpp', '-Wno-unused-function', '-std=c99']

ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir={'pycocotools': 'pycocotools'},
    version='2.0',
    ext_modules=ext_modules
)