from setuptools import setup, Extension
import subprocess
import os
try:
    import numpy
except (ImportError, ModuleNotFoundError):
    raise ImportError('This package requires "numpy" to be installed. '
                      'Install it first: "pip install numpy".')

mock_install = os.environ.get('READTHEDOCS') == 'True'

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    from Cython.Compiler import Options
except ImportError:
    USE_CYTHON = False
    compiler_directives = {}
else:
    compiler_directives = Options.get_directive_defaults()
    USE_CYTHON = True

compiler_directives["emit_code_comments"] = False

ext = '.pyx' if USE_CYTHON else '.c'


# ------------------------------------------
# Get include and lib directories of libgsl
# These are taken and adapted from
# https://github.com/twiecki/CythonGSL
# so we don't need to import cython_gsl here
# ------------------------------------------

def get_gsl_include_dir():
    try:
        gsl_include = subprocess.check_output(
            'gsl-config --cflags', shell=True).decode('utf-8')[2:-1]
    except subprocess.CalledProcessError:
        gsl_include = os.getenv('LIB_GSL')
        if gsl_include is not None:
            gsl_include += '/include'
    if gsl_include == '' or gsl_include is None:
        raise ImportError("Could not find libgsl.")
    return gsl_include


def get_gsl_lib_dir():
    try:
        lib_gsl_dir = subprocess.check_output(
            'gsl-config --libs', shell=True).decode('utf-8').split()[0][2:]
    except subprocess.CalledProcessError:
        lib_gsl_dir = os.getenv('LIB_GSL')
        if lib_gsl_dir is not None:
            lib_gsl_dir += '/lib'
    if lib_gsl_dir == '' or lib_gsl_dir is None:
        raise ImportError("Could not find libgsl.")
    return lib_gsl_dir


# Check if libgsl is available
try:
    GSL_INC_DIR = get_gsl_include_dir()
    GSL_LIB_DIR = get_gsl_lib_dir()
    GSL = True
except ImportError:
    GSL = False

change_libraries = ['gsl', 'gslcblas']
if mock_install or not GSL:
    change_library_dirs = []
else:
    change_library_dirs = [GSL_LIB_DIR]
change_include_dirs = ['.']

cmdclass = {}

extensions = [
    Extension("nd._filters", ["nd/_filters" + ext],
              extra_compile_args=['-O3'],
              extra_link_args=[],
              ),
    Extension("nd._warp", ["nd/_warp" + ext]),
]
if GSL:
    # Only build the cython change module
    # if libgsl is available
    extensions.append(
        Extension("nd._change", ["nd/_change" + ext],
                  libraries=change_libraries,
                  library_dirs=change_library_dirs,
                  include_dirs=change_include_dirs,
                  extra_compile_args=['-O3'],
                  extra_link_args=[],
                  )
    )

if USE_CYTHON:
    extensions = cythonize(
        extensions, compiler_directives=compiler_directives)
    cmdclass = {'build_ext': build_ext}

if mock_install:
    include_dirs = []
elif not GSL:
    include_dirs = [
        numpy.get_include()
    ]
else:
    include_dirs = [
        numpy.get_include(),
        GSL_INC_DIR
    ]

install_requires = [
    "numpy",
    "xarray",
    "dask[dataframe]",
    "pandas",
    "python-dateutil",
    "matplotlib",
]

if not mock_install:
    install_requires.extend([
        "scipy",
        "lxml",
        "rasterio>=1.0.13",
        "affine",
        "opencv-python",
        # "NetCDF4"
        "h5py",
        "h5netcdf",
        "imageio",
        "imageio-ffmpeg",
        "pyproj>=2.0",
        "geopandas",
        "scikit-image",
        "multiprocess",
    ])

setup(
    use_scm_version=True,
    cmdclass=cmdclass,
    ext_modules=extensions if not mock_install else [],
    include_dirs=include_dirs,
    install_requires=install_requires
)
