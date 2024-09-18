import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='',
    ext_modules=[
        CUDAExtension('ModuleMapGraph', [
            'GPU/SRC/ModuleMapGraph.cc',
            'GPU/SRC/module_map_doublet.cu',
            'GPU/SRC/module_map_triplet.cu',
            'GPU/SRC/TTree_hits.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })