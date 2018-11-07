from setuptools import setup, Extension
import subprocess
import numpy
import cython_gsl

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

ext = '.pyx' if use_cython else '.c'

# from Cython.Compiler.Options import get_directive_defaults
# directive_defaults = get_directive_defaults()
# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True

extensions = [
    Extension("geo.change._omnibus", ["geo/change/_omnibus" + ext],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              include_dirs=['.', cython_gsl.get_cython_include_dir()],
              extra_compile_args=['-O3', '-fopenmp'],
              extra_link_args=['-fopenmp'],
              ),
    Extension("geo.filter._nlmeans", ["geo/filter/_nlmeans" + ext],
              extra_compile_args=['-O3', '-fopenmp'],
              extra_link_args=['-fopenmp'],
              ),
    Extension("geo._warp", ["geo/_warp" + ext]),
]

if use_cython:
    extensions = cythonize(extensions)
    cmdclass = {'build_ext': build_ext}
else:
    cmdclass = {}


gdal_version = subprocess.check_output(
    ['gdal-config', '--version']).decode('utf-8').strip('\n')
gdal_version_range = [gdal_version + '.0', gdal_version + '.999']

setup(
    cmdclass=cmdclass,
    ext_modules=extensions,
    include_dirs=[
        numpy.get_include(),
        cython_gsl.get_include()
    ],
    install_requires=[
        "numpy",
        "scipy",
        "xarray",
        "dask[dataframe]",
        "lxml",
        "pygdal>={},<={}".format(*gdal_version_range),
        "pandas",
        "python-dateutil",
        "matplotlib",
        "affine",
        "opencv-python",
        "NetCDF4"
    ]
)
