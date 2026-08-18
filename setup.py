import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        cmake_version = tuple(int(p) for p in re.search(r'version\s*([\d.]+)', out.decode()).group(1).split('.')[:3])
        if cmake_version < (3, 15):
            raise RuntimeError("CMake >= 3.15 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='directml',
    version='1.0.0',
    author='Microsoft Corporation',
    author_email='askdirectml@microsoft.com',
    description='Python Binding for DirectML Samples',
    long_description='PyDirectML is a small Python binding library for DirectML written to facilitate DirectML sample authoring in Python. It simplifies DirectML graph authoring and execution with automatic resource management and binding support through NumPy arrays.',
    url="https://github.com/aisk/PyDirectML",
    license='MIT',
    python_requires='>=3.6',
    ext_modules=[CMakeExtension('directml')],
    cmdclass=dict(build_ext=CMakeBuild),
    keywords='DirectML Python samples',
    setup_requires=['cmake'],
    zip_safe=False
)
