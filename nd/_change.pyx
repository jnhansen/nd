cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from scipy.stats import chi2
from libc.math cimport abs, log, isnan
# from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython cimport floating
cimport cython_gsl


ctypedef np.float64_t DOUBLE
ctypedef np.float32_t FLOAT
ctypedef Py_ssize_t SIZE_TYPE
ctypedef unsigned char BOOL

# =================================================================

@cython.cdivision(True)
cdef double _f(double p, double k, double n) nogil:
    cdef double f
    f = (k - 1) * p**2
    return f


@cython.cdivision(True)
cdef double _rho(double p, double k, double n) nogil:
    cdef double rho
    rho = 1 - (2 * p**2 - 1) / (6 * (k - 1) * p) * (k/n - 1/(n*k))
    return rho


@cython.cdivision(True)
cdef double _omega2(double p, double k, double n, double rho) nogil:
    cdef double omega2
    omega2 = p**2 * (p**2 - 1) / (24 * rho**2) \
        * (k/(n**2) - 1/((n*k)**2)) \
        - p**2 * (k - 1) / 4 * (1 - 1/rho)**2
    return omega2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# cpdef floating _z(np.ndarray[floating, ndim=2] ts, unsigned int n):
cpdef floating _z(floating [:, :] ts, unsigned int n) nogil:
    """
    The 4 columns in ts are [C11, C12.real, C21.imag, C22]
    """
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



#
# This function is deprecated and should not be used.
#
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[floating, ndim=2] array_omnibus(floating [:, :, :, :] ts,
                                                 unsigned int n):
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
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef floating single_pixel_omnibus(floating [:, :] ts, unsigned int n) nogil:
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
    _P1 = cython_gsl.gsl_cdf_chisq_P(z, f)
    _P2 = cython_gsl.gsl_cdf_chisq_P(z, f+4)

    result = _P1 + omega2 * (_P2 - _P1)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[BOOL, ndim=1] change_list_to_bool(np.ndarray[floating, ndim=1] c):
    """Given an array of changepoint indices as generated from the change detection method,
    extract a boolean array that is True where a valid change was detected
    and False everywhere else.
    """
    cdef:
        size_t k = c.shape[0]
        size_t last_change = 0
        size_t idx
        size_t i
        np.ndarray[BOOL, ndim=1] res = np.zeros(k, dtype=np.uint8)

    for i in range(k):
        idx = int(c[i])
        if not isnan(c[i]) and idx > last_change:
            res[idx] = 1
            last_change = idx

    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[BOOL, ndim=3] change_array_to_bool(np.ndarray[floating, ndim=3] c):
    """Given an array of changepoint indices as generated from the change detection method,
    extract a boolean array that is True where a valid change was detected
    and False everywhere else.

    NOTE: Cython doesn't yet support boolean arrays, so the output is char and
    should be converted to bool later if desired.

    Parameters
    ----------
    c : np.ndarray, shape (T, M, N)
        The detected change indices. The first dimension is time.

    Returns
    -------
    np.ndarray, shape (T, M, N)
        The boolean array.
    """
    cdef:
        size_t k = c.shape[0]
        size_t nrows = c.shape[1]
        size_t ncols = c.shape[2]
        size_t t, i, j
        size_t idx, last_change
        np.ndarray[BOOL, ndim=3] res = np.zeros((k, nrows, ncols), dtype=np.uint8)
        floating _pixel

    for i in range(nrows):
        for j in range(ncols):
            last_change = 0
            for t in range(k):
                _pixel = c[t, i, j]
                idx = int(_pixel)
                if not isnan(_pixel) and idx > last_change:
                    res[idx, i, j] = 1
                    last_change = idx

    return res


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef void single_pixel_change_detection(floating [:, :] ts,
                                         BOOL [:] result,
                                         double alpha,
                                         unsigned int n) nogil:
    cdef:
        floating [:, :] subset
        SIZE_TYPE k = ts.shape[0]
        SIZE_TYPE l, j, r
        floating p_H0_l, p_H0_lj
        BOOL _change

    l = 0
    while True:
        # Test global hypotheses H0_l
        subset = ts[l:, :]
        p_H0_l = single_pixel_omnibus(subset, n=n)
        _change = (p_H0_l > alpha)
        if not _change:
            break
        else:
            # Test marginal hypotheses
            # j is the number of time points to consider in the omnibus tests
            for j in range(2, k - l + 1):
                subset = ts[l:l+j, :]
                p_H0_lj = single_pixel_omnibus(subset, n=n)
                _change = (p_H0_lj > alpha)
                # Break on first significant change
                r = j - 1
                if _change:
                    result[l + r] = 1
                    break
        l = l + r
        if l >= k - 1:
            break


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef BOOL [:, :, :] change_detection(floating [:, :, :, :] values,
                                      double alpha, unsigned int n=1,
                                      unsigned int njobs=1):
    """
    `ds` is already multilooked (with `n` looks).
    """
    cdef:
        SIZE_TYPE nrows = values.shape[0]
        SIZE_TYPE ncols = values.shape[1]
        SIZE_TYPE k = values.shape[2]
        SIZE_TYPE i_lat, i_lon
        floating [:, :] ts
        BOOL [:, :, :] result = np.zeros((nrows, ncols, k), dtype=np.uint8,
                                         order='C')
        unsigned int num_threads = njobs

    # Do change detection completely independently for each pixel.
    for i_lat in prange(nrows, nogil=True, schedule='dynamic', chunksize=100,
                        num_threads=num_threads):
        for i_lon in range(ncols):
            single_pixel_change_detection(values[i_lat, i_lon, :, ::1],
                                          result[i_lat, i_lon, :],
                                          alpha=alpha, n=n)

    return result
