from setuptools import setup, find_packages, Extension
import subprocess
from Cython.Distutils import build_ext
import numpy
import cython_gsl

# from Cython.Compiler.Options import get_directive_defaults
# directive_defaults = get_directive_defaults()
# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True

ext_modules = [
    Extension("geo.change._omnibus", ["geo/change/_omnibus.pyx"],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              include_dirs=['.', cython_gsl.get_cython_include_dir()],
              extra_compile_args=['-O3', '-fopenmp'],
              extra_link_args=['-fopenmp'],
              ),
    Extension("geo.filter._nlmeans", ["geo/filter/_nlmeans.pyx"],
              extra_compile_args=['-O3', '-fopenmp'],
              extra_link_args=['-fopenmp'],
              ),
    Extension("geo._warp", ["geo/_warp.pyx"]),
]

gdal_version = subprocess.check_output(
    ['gdal-config', '--version']).decode('utf-8').strip('\n')
gdal_version_range = [gdal_version + '.0', gdal_version + '.999']

setup(
    name='geotools',
    packages=find_packages(),
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[
        numpy.get_include(),
        cython_gsl.get_include()
    ],
    install_requires=[
        # "gcc",
        "numpy",
        "scipy",
        "xarray",
        "dask",
        "cython",
        "lxml",
        "pygdal>={},<={}".format(*gdal_version_range),
        "pandas",
        "python-dateutil",
        "scikit-image",
        "matplotlib",
        "affine",
        "CythonGSL",
        "opencv-python"
    ]
)
