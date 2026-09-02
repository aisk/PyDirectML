"""Build the _core extension with CMake. Everything else is in pyproject.toml."""

import os
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # Have CMake write the .pyd where setuptools expects the extension: the
        # package directory for build_ext --inplace, the build tree otherwise.
        output_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        source_dir = os.path.dirname(os.path.abspath(__file__))
        config = "Debug" if self.debug else "Release"

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call([
            "cmake", source_dir,
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={output_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ], cwd=self.build_temp)
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", config, "--parallel"],
            cwd=self.build_temp)


setup(ext_modules=[CMakeExtension("directml._core")], cmdclass={"build_ext": CMakeBuild})
