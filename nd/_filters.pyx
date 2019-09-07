cimport cython
from cython.parallel import prange
from cython cimport floating
import numpy as np
cimport numpy as np
from libc.math cimport abs, log, isnan, exp, sqrt


ctypedef Py_ssize_t SIZE_TYPE

cdef short EDGE_MODE_REFLECT = 0
cdef short EDGE_MODE_REPEAT = 1


cdef Py_ssize_t _idx(Py_ssize_t i, Py_ssize_t shape,
                     short mode=EDGE_MODE_REFLECT):
    """
    Sanitize index i given the shape of its dimension
    by reflecting at the boundary.

    Parameters
    ----------
    mode : short
        one of EDGE_MODE_REPEAT or EDGE_MODE_REFLECT
    """
    if mode == EDGE_MODE_REPEAT:
        if i < 0:
            return 0
        elif i >= shape:
            return shape - 1
        else:
            return i

    elif mode == EDGE_MODE_REFLECT:
        if i < 0:
            return -i
        elif i >= shape:
            return 2*shape - 2 - i
        else:
            return i


# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# # @cython.cdivision(True)
# cpdef void _patchwise_nlmeans(np.ndarray[floating, ndim=3] arr,
#                               np.ndarray[floating, ndim=3] output,
#                               unsigned int r, unsigned int f,
#                               double sigma, double h):
#     cdef:
#         SIZE_TYPE nrows = arr.shape[0]
#         SIZE_TYPE ncols = arr.shape[1]
#         SIZE_TYPE nvars = arr.shape[2]
#         SIZE_TYPE ndims = 2
#         SIZE_TYPE i, j, v
#         SIZE_TYPE ip, jp, iq, jq
#         int di, dj
#         double total_weight, max_weight, weight, dsquare
#         np.ndarray[floating, ndim=3] weighted_sum
#         short m = EDGE_MODE_REFLECT

#     # Make sure output is zero before we start.
#     for i in range(nrows):
#         for j in range(ncols):
#             for v in range(nvars):
#                 output[i, j, v] = 0

#     for ip in range(0, nrows):
#         for jp in range(0, ncols):
#             #
#             # (ip, jp) is now the center of patch P
#             #

#             # Initialize weights
#             total_weight = 0
#             max_weight = 0
#             weighted_sum = np.zeros((2*r+1, 2*r+1, nvars))

#             # Loop through all patches Q in neighborhood of P
#             for iq in range(ip - r, ip + r + 1):
#                 for jq in range(jp - r, jp + r + 1):
#                     # Exclude p == q for now.
#                     if (ip == iq) and (jp == jq):
#                         continue

#                     #
#                     # (iq, jq) is now the center of patch Q
#                     #
#                     dsquare = 0
#                     for di in range(-f, f+1):
#                         for dj in range(-f, f+1):
#                             for v in range(nvars):
#                                 dsquare += (
#                                     arr[_idx(ip + di, nrows, m),
#                                         _idx(jp + dj, ncols, m), v] -
#                                     arr[_idx(iq + di, nrows, m),
#                                         _idx(jq + dj, ncols, m), v]
#                                 ) ** 2


#                     dsquare /= nvars * (2*f + 1)**ndims

#                     weight = exp( -max(dsquare - 2*sigma**2, 0) / h**2 )

#                     total_weight += weight
#                     if weight > max_weight:
#                         max_weight = weight

#                     for i in range(iq - f, iq + f + 1):
#                         for j in range(jq - f, jq + f + 1):
#                             for v in range(nvars):
#                                 weighted_sum[i - iq + f, j - jq + f, v] += \
#                                     weight * arr[_idx(i, nrows, m),
#                                                  _idx(j, ncols, m), v]

#             # Now we have the weighted sum w(Bp, Bq)

#             # Include pixel itself
#             # And assign to output pixel
#             total_weight += max_weight

#             for i in range(ip - r, ip + r + 1):
#                 for j in range(jp - r, jp + r + 1):
#                     for v in range(nvars):
#                         # weighted_sum[i-iP_min, j-jP_min, v] += \
#                         #     max_weight * arr[i, j, v]
#                         # output[i, j, v] = weighted_sum[i, j, v] / total_weight
#                         output[_idx(i, nrows, m), _idx(j, ncols, m), v] += (
#                             weighted_sum[i - ip + r, j - jp + r, v] + \
#                             max_weight * \
#                             arr[_idx(i, nrows, m), _idx(j, ncols, m), v]
#                         ) / total_weight
#                         # output[i, j, v] = \
#                         #     (weighted_sum[i-iP_min, j-jP_min, v] + \
#                         #     max_weight * arr[i, j, v]) / total_weight

#     # Need to renormalize every output pixel by the number of times it has been
#     # computed.
#     for i in range(nrows):
#         for j in range(ncols):
#             for v in range(nvars):
#                 output[i, j, v] /= (2*r + 1)**2



# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cpdef void _pixelwise_nlmeans(floating [:, :, :] arr,
#                               floating [:, :, :] output,
#                               unsigned int r, unsigned int f,
#                               double sigma, double h):
#     cdef:
#         SIZE_TYPE nrows = arr.shape[0]
#         SIZE_TYPE ncols = arr.shape[1]
#         SIZE_TYPE nvars = arr.shape[2]
#         SIZE_TYPE i, j, v, ip, jp, iq, jq
#         SIZE_TYPE di, dj
#         double total_weight, max_weight, weight, dsquare
#         floating [:] weighted_sum = np.zeros(nvars)
#         SIZE_TYPE ndims = 2
#         short m = EDGE_MODE_REFLECT

#     # Make sure output is zero before we start.
#     for i in range(nrows):
#         for j in range(ncols):
#             for v in range(nvars):
#                 output[i, j, v] = 0

#     # Loop through all pixels p in the image.
#     for ip in range(nrows):
#         for jp in range(ncols):
#             # Initialize all weights to 0
#             total_weight = 0
#             max_weight = 0
#             for v in range(nvars):
#                 weighted_sum[v] = 0

#             # For the pixel p = (ip, jp), compute the weight
#             # of every other pixel in the neighborhood B(p, r).
#             for iq in range(ip - r, ip + r + 1):
#                 for jq in range(jp - r, jp + r + 1):
#                     # Compute the weight between pixel p = (ip, jp)
#                     # and pixel q = (iq, jq)

#                     # Exclude p == q for now.
#                     if (ip == iq) and (jp == jq):
#                         continue

#                     dsquare = 0
#                     for di in range(-f, f + 1):
#                         for dj in range(-f, f + 1):
#                             for v in range(nvars):
#                                 dsquare += (
#                                     arr[_idx(ip + di, nrows, m),
#                                         _idx(jp + dj, ncols, m), v] -
#                                     arr[_idx(iq + di, nrows, m),
#                                         _idx(jq + dj, ncols, m), v]
#                                 ) ** 2

#                     dsquare /= nvars * (2*f + 1)**ndims

#                     # Update weights
#                     weight = exp( -max(dsquare - 2*sigma**2, 0) / h**2 )

#                     total_weight += weight
#                     if weight > max_weight:
#                         max_weight = weight

#                     for v in range(nvars):
#                         weighted_sum[v] += weight * arr[_idx(iq, nrows, m),
#                                                         _idx(jq, ncols, m), v]

#             # Include pixel itself
#             # And assign to output pixel
#             total_weight += max_weight
#             for v in range(nvars):
#                 weighted_sum[v] += max_weight * arr[ip, jp, v]
#                 output[ip, jp, v] = weighted_sum[v] / total_weight



# cpdef void _single_pixel_nlmeans(floating [:, :, :, :] arr,
#                                  floating [:, :, :, :] output,
#                                  unsigned int [:] r,
#                                  unsigned int [:] f,
#                                  double sigma, double h):
#     cdef:
#         SIZE_TYPE ndims = 3
#         SIZE_TYPE [3] N = [arr.shape[0], arr.shape[1], arr.shape[2]]
#         SIZE_TYPE nvars = arr.shape[3]
#         SIZE_TYPE [3] i, p, q, d
#         floating total_weight, max_weight, weight, dsquare
#         floating [:] weighted_sum
#         floating dsq_norm = nvars * (2*f[0] + 1) * (2*f[1] + 1) * (2*f[2] + 1)
#         short m = EDGE_MODE_REFLECT

#     # Initialize all weights to 0
#     total_weight = 0
#     max_weight = 0
#     for v in range(nvars):
#         weighted_sum[v] = 0

#     # For the pixel p, compute the weight
#     # of every other pixel in the neighborhood B(p, r).
#     for q[0] in range(p[0] - r[0], p[0] + r[0] + 1):
#         for q[1] in range(p[1] - r[1], p[1] + r[1] + 1):
#             for q[2] in range(p[2] - r[2], p[2] + r[2] + 1):
#                 # Compute the weight between pixel p and q

#                 # Exclude p == q for now.
#                 if (p[0] == q[0]) and (p[1] == q[1]) and (p[2] == q[2]):
#                     continue

#                 dsquare = 0
#                 for d[0] in range(-f[0], f[0] + 1):
#                     for d[1] in range(-f[1], f[1] + 1):
#                         for d[2] in range(-f[2], f[2] + 1):
#                             for v in range(nvars):
#                                 dsquare += (
#                                     arr[_idx(p[0] + d[0], N[0], m),
#                                         _idx(p[1] + d[1], N[1], m),
#                                         _idx(p[2] + d[2], N[2], m),
#                                         v] -
#                                     arr[_idx(q[0] + d[0], N[0], m),
#                                         _idx(q[1] + d[1], N[1], m),
#                                         _idx(q[2] + d[2], N[2], m),
#                                         v]
#                                 ) ** 2

#                 dsquare /= dsq_norm

#                 # Update weights
#                 weight = exp( -max(dsquare - 2*sigma**2, 0) / h**2 )

#                 total_weight += weight
#                 if weight > max_weight:
#                     max_weight = weight

#                 for v in range(nvars):
#                     weighted_sum[v] += weight * \
#                         arr[_idx(q[0], N[0], m),
#                             _idx(q[1], N[1], m),
#                             _idx(q[2], N[2], m), v]

#     if max_weight == 0:
#         max_weight = 1

#     # Include pixel itself
#     # And assign to output pixel
#     total_weight += max_weight
#     for v in range(nvars):
#         weighted_sum[v] += max_weight * arr[p[0], p[1], p[2], v]
#         output[p[0], p[1], p[2], v] = weighted_sum[v] / total_weight


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double find_weight(double weight_sum,
                         double sq_weight_sum,
                         double n):
    """
    Find additional weight w such that
    the effective sample size equals n.
    """
    cdef:
        double rt
        Py_ssize_t i

    if n - 1 > weight_sum**2 / sq_weight_sum:
        raise ValueError('No solution')

    rt = sqrt(n*weight_sum*weight_sum - n*n*sq_weight_sum + n*sq_weight_sum)
    return (weight_sum + rt)/(n-1)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void _pixelwise_nlmeans_3d(floating [:, :, :, :] arr,
                                 floating [:, :, :, :] output,
                                 unsigned int [:] r,
                                 unsigned int [:] f,
                                 double sigma, double h,
                                 double n_eff=-1):
    """
    n_eff = -1 means no fixed effective sample size
    """
    cdef:
        SIZE_TYPE ndims = 3
        SIZE_TYPE [3] N = [arr.shape[0], arr.shape[1], arr.shape[2]]
        SIZE_TYPE nvars = arr.shape[3]
        SIZE_TYPE [3] i, p, q, d
        double total_weight, total_sq_weight,
        double max_weight, weight, dsquare
        floating [:] weighted_sum
        floating dsq_norm = nvars * (2*f[0] + 1) * (2*f[1] + 1) * (2*f[2] + 1)
        short m = EDGE_MODE_REFLECT

    dtype = np.float32 if floating is float else np.float64
    weighted_sum = np.zeros(nvars, dtype=dtype)

    # Make sure output is zero before we start.
    for i[0] in range(N[0]):
        for i[1] in range(N[1]):
            for i[2] in range(N[2]):
                for v in range(nvars):
                    output[i[0], i[1], i[2], v] = 0

    # Loop through all pixels p in the image.
    for p[0] in range(N[0]):
        for p[1] in range(N[1]):
            for p[2] in range(N[2]):
                # Initialize all weights to 0
                total_weight = 0
                total_sq_weight = 0
                max_weight = 0
                for v in range(nvars):
                    weighted_sum[v] = 0

                # For the pixel p, compute the weight
                # of every other pixel in the neighborhood B(p, r).
                for q[0] in range(p[0] - r[0], p[0] + r[0] + 1):
                    for q[1] in range(p[1] - r[1], p[1] + r[1] + 1):
                        for q[2] in range(p[2] - r[2], p[2] + r[2] + 1):
                            # Compute the weight between pixel p and q

                            # Exclude p == q for now.
                            if (p[0] == q[0]) and (p[1] == q[1]) and (p[2] == q[2]):
                                continue

                            dsquare = 0
                            for d[0] in range(-f[0], f[0] + 1):
                                for d[1] in range(-f[1], f[1] + 1):
                                    for d[2] in range(-f[2], f[2] + 1):
                                        for v in range(nvars):
                                            dsquare += (
                                                arr[_idx(p[0] + d[0], N[0], m),
                                                    _idx(p[1] + d[1], N[1], m),
                                                    _idx(p[2] + d[2], N[2], m),
                                                    v] -
                                                arr[_idx(q[0] + d[0], N[0], m),
                                                    _idx(q[1] + d[1], N[1], m),
                                                    _idx(q[2] + d[2], N[2], m),
                                                    v]
                                            ) ** 2

                            dsquare /= dsq_norm

                            # Update weights
                            weight = exp( -max(dsquare - 2*sigma**2, 0) / h**2 )

                            total_weight += weight
                            total_sq_weight += weight*weight

                            if weight > max_weight:
                                max_weight = weight

                            for v in range(nvars):
                                weighted_sum[v] += weight * \
                                    arr[_idx(q[0], N[0], m),
                                        _idx(q[1], N[1], m),
                                        _idx(q[2], N[2], m), v]

                # Determine weight of pixel itself
                if n_eff < 0:
                    # If no effective sample size is given, assign maximum
                    # weight found
                    if max_weight == 0:
                        max_weight = 1
                    weight = max_weight
                else:
                    weight = find_weight(total_weight, total_sq_weight, n_eff)

                # Include pixel itself
                # And assign to output pixel
                total_weight += weight
                for v in range(nvars):
                    weighted_sum[v] += weight * arr[p[0], p[1], p[2], v]
                    output[p[0], p[1], p[2], v] = weighted_sum[v] / total_weight
