import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='module_map',
    ext_modules=[
        CUDAExtension('module_map', [
            'module_map.cc',
            'module_map_doublet.cu',
            'module_map_triplet.cu',
            'TTree_hits.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })