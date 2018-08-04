from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy


ext_modules = [Extension("geo._omnibus", ["geo/_omnibus.pyx"],)]

setup(name='geotools',
      packages=find_packages(),
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules,
      include_dirs=[numpy.get_include()],
      install_requires=[
        "numpy",
        "scipy",
        "xarray",
        "dask",
        "cython",
        "lxml",
        "gdal",
        "pandas",
        "python-dateutil",
        "scikit-image",
        "matplotlib"
      ])
