import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

modules = [
    CppExtension(
        'roi_align.crop_and_resize_cpu',
        ['roi_align/src/crop_and_resize.cpp']
        )
]

setup(
    name='roi_align',
    version='0.0.1',
    description='PyTorch version of RoIAlign',
    author='Long Chen',
    author_email='longch1024@gmail.com',
    url='https://github.com/longcw/RoIAlign.pytorch',
    packages=find_packages(exclude=('tests',)),

    ext_modules=modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)
