cimport cython
import numpy as np
cimport numpy as np
from scipy.stats import chi2
from libc.math cimport abs, log
from cython cimport floating

ctypedef np.float64_t DOUBLE
ctypedef np.float32_t FLOAT

# =================================================================

@cython.cdivision(True)
cdef double _f(double p, double k, double n):
    cdef double f
    f = (k - 1) * p**2
    return f


@cython.cdivision(True)
cdef double _rho(double p, double k, double n):
    cdef double rho
    rho = 1 - (2 * p**2 - 1) / (6 * (k - 1) * p) * (k/n - 1/(n*k))
    return rho

    
@cython.cdivision(True)
cdef double _omega2(double p, double k, double n, double rho):
    cdef double omega2
    omega2 = p**2 * (p**2 - 1) / (24 * rho**2) \
        * (k/(n**2) - 1/((n*k)**2)) \
        - p**2 * (k - 1) / 4 * (1 - 1/rho)**2
    return omega2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# cpdef floating _z(np.ndarray[floating, ndim=2] ts, unsigned int n):
cpdef floating _z(floating [:, :] ts, unsigned int n):
    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef:
        floating p = 2,             # dual pol: p=2, full pol: p=3
        size_t k = ts.shape[0],     # number of matrices (time steps)
        floating c11sum = 0, c22sum = 0, c12rsum = 0, c12isum = 0
        floating det_of_sum
        DOUBLE prod_of_dets = 1.0
        floating rho
        DOUBLE logQ
        floating z
        floating prob
        size_t i
    
    # Compute the individual matrix determinants and the sum along the
    # time dimension.
    for i in range(k):
        prod_of_dets *= ((ts[i, 0] * ts[i, 3]) - (ts[i, 1]**2 + ts[i, 2]**2))        
        c11sum += ts[i, 0]
        c12rsum += ts[i, 1]
        c12isum += ts[i, 2]
        c22sum += ts[i, 3]
    
    # The determinant of the sum of all matrices.
    det_of_sum = ((c11sum * c22sum) - (c12rsum**2 + c12isum**2))

    logQ = n * (p*k*log(k) + log(prod_of_dets) - k*log(det_of_sum))
    rho = _rho(p, k, n)
    z = -2 * rho * logQ
    return z



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# cpdef np.ndarray[floating, ndim=2] array_omnibus(np.ndarray[floating, ndim=4] ts, unsigned int n):
cpdef np.ndarray[floating, ndim=2] array_omnibus(floating [:, :, :, :] ts, unsigned int n):
    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef:
        size_t k = ts.shape[0]
        size_t nrows = ts.shape[1]
        size_t ncols = ts.shape[2]
        size_t i, j
        double p = 2
        double f, rho, omega2
        
        # np.ndarray[floating, ndim=2] _single_ts = np.empty((k, 4), dtype=dtype)
        floating [:, :] _single_ts # = np.empty((k, 4), dtype=dtype)
        # np.ndarray[floating, ndim=2] zs = np.empty((nrows, ncols), dtype=dtype)
        floating [:, :] zs = np.empty((nrows, ncols), dtype=dtype)
        np.ndarray[floating, ndim=2] _P1 # = np.zeros((nrows, ncols), dtype=np.float64)
        # floating [:, :] _P1 # = np.zeros((nrows, ncols), dtype=np.float64)
        np.ndarray[floating, ndim=2] _P2 # = np.zeros((nrows, ncols), dtype=np.float64)
        # floating [:, :] _P2 # = np.zeros((nrows, ncols), dtype=np.float64)
        np.ndarray[floating, ndim=2] result = np.empty((nrows, ncols), dtype=dtype)
        # floating [:, :] result = np.empty((nrows, ncols), dtype=dtype)
    
    for i in range(nrows):
        for j in range(ncols):
            _single_ts = ts[:, i, j, :]
            zs[i, j] = _z(_single_ts, n=n)

    f = _f(p, k, n)
    rho = _rho(p, k, n)
    omega2 = _omega2(p, k, n, rho)

    # chi2.cdf always returns 64bit float ...
    _P1 = chi2.cdf(zs, f).astype(dtype)
    _P2 = chi2.cdf(zs, f+4).astype(dtype)

    result = _P1 + omega2 * (_P2 - _P1)
    return result


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
# cpdef floating single_pixel_omnibus(np.ndarray[floating, ndim=2] ts, unsigned int n):
cpdef floating single_pixel_omnibus(floating [:, :] ts, unsigned int n):
    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef:
        double p = 2
        size_t k = ts.shape[0]
        double f, rho, omega2
        double prob
        floating z
        floating _P1, _P2
        floating result
    
    f = _f(p, k, n)
    rho = _rho(p, k, n)
    omega2 = _omega2(p, k, n, rho)
    z = _z(ts, n)
    _P1 = chi2.cdf(z, f).astype(dtype)
    _P2 = chi2.cdf(z, f+4).astype(dtype)

    result = _P1 + omega2 * (_P2 - _P1)
    return result
