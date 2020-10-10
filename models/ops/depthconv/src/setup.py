from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

from distutils.command.build_ext import build_ext
from distutils.sysconfig import customize_compiler

class my_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)

this_file = os.path.dirname(__file__)

sources = ['src/depthconv.cpp' , 'src/depthconv_cuda.cpp']
# headers = ['src/depthavgpooling.h', 'src/depthavgpooling_cuda.h']
defines = [('WITH_CUDA', None)]
# with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))

print(this_file)
extra_objects = ['depthconv_cuda_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]



setup(
    name='depthconv',
    ext_modules=[
        CppExtension(
            name='depthconv',
            sources = sources,
            # define_macros=defines,
            extra_objects=extra_objects,
            extra_compile_args=["-g"],
        )
    ],
    cmdclass={
        'build_ext': my_build_ext
})
