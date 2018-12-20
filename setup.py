from setuptools import setup, Extension
import subprocess
import os
try:
    import numpy
except (ImportError, ModuleNotFoundError):
    raise ImportError('This package requires "numpy" to be installed. '
                      'Install it first: "pip install numpy".')
try:
    import cython_gsl
except (ImportError, ModuleNotFoundError):
    raise ImportError('This package requires "cythongsl" to be installed. '
                      'Install it first: "pip install cythongsl".')

mock_install = os.environ.get('READTHEDOCS') == 'True'

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

omnibus_libraries = []
omnibus_library_dirs = []
omnibus_include_dirs = ['.']

if use_cython:
    omnibus_libraries.extend(cython_gsl.get_libraries())
    omnibus_library_dirs.append(cython_gsl.get_library_dir())
    omnibus_include_dirs.append(cython_gsl.get_cython_include_dir())


cmdclass = {}

extensions = [
    Extension("nd.change._omnibus", ["nd/change/_omnibus" + ext],
              libraries=omnibus_libraries,
              library_dirs=omnibus_library_dirs,
              include_dirs=omnibus_include_dirs,
              extra_compile_args=['-O3', '-fopenmp'],
              extra_link_args=['-fopenmp'],
              ),
    Extension("nd.filters._nlmeans", ["nd/filters/_nlmeans" + ext],
              extra_compile_args=['-O3', '-fopenmp'],
              extra_link_args=['-fopenmp'],
              ),
    Extension("nd.filters._correlation", ["nd/filters/_correlation" + ext]),
    Extension("nd.warp._warp", ["nd/warp/_warp" + ext]),
]

if use_cython:
    extensions = cythonize(extensions)
    cmdclass = {'build_ext': build_ext}

include_dirs = []
install_requires = []

if not mock_install:
    install_requires.extend([
        "numpy",
        "scipy",
        "xarray",
        "dask[dataframe]",
        "lxml",
        "rasterio",
        "pandas",
        "python-dateutil",
        "matplotlib",
        "affine",
        "opencv-python",
        # "NetCDF4"
        "h5py",
        "h5netcdf",
        "imageio",
        "pyproj",
        "geopandas"
    ])

include_dirs.append(numpy.get_include())
include_dirs.append(cython_gsl.get_include())

setup(
    cmdclass=cmdclass,
    ext_modules=extensions,
    include_dirs=include_dirs,
    install_requires=install_requires,
    dependency_links=[
        "https://github.com/jswhit/pyproj.git#egg=pyproj"
    ]
)
