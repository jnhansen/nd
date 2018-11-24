from setuptools import setup, Extension
import subprocess

try:
    import numpy
    import cython_gsl
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    cython_gsl.get_library_dir()
except IndexError:
    # IndexError is raised if the gsl library is not installed.
    # For building the documentation this is not required.
    use_cython = False
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

extensions = [
    Extension("nd.change._omnibus", ["nd/change/_omnibus" + ext],
              libraries=omnibus_libraries,
              library_dirs=omnibus_library_dirs,
              include_dirs=omnibus_include_dirs,
              extra_compile_args=['-O3', '-fopenmp'],
              extra_link_args=['-fopenmp'],
              ),
    Extension("nd.filter._nlmeans", ["nd/filter/_nlmeans" + ext],
              extra_compile_args=['-O3', '-fopenmp'],
              extra_link_args=['-fopenmp'],
              ),
    Extension("nd._warp", ["nd/_warp" + ext]),
]

if use_cython:
    extensions = cythonize(extensions)
    cmdclass = {'build_ext': build_ext}
else:
    cmdclass = {}

try:
    gdal_version = subprocess.check_output(
        ['gdal-config', '--version']).decode('utf-8').strip('\n')
    gdal_version_range = [gdal_version + '.0', gdal_version + '.999']
except FileNotFoundError:
    # gdal is not installed.
    # Assume this is a mock install.
    gdal_version_range = [0, 99]

include_dirs = []
if use_cython:
    include_dirs.append(numpy.get_include())
    include_dirs.append(cython_gsl.get_include())
setup(
    cmdclass=cmdclass,
    ext_modules=extensions,
    include_dirs=include_dirs,
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
