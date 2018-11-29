.. _omnibus:

=============================
Omnibus Test Change Detection
=============================

.. currentmodule:: geo.change

Conradsen et al. (2016) present a change detection algorithm for time series of complex valued SAR data based on the complex Wishart distribution for the covariance matrices.
:math:`S_{rt}` denotes the complex scattering amplitude where :math:`r,t \in \{h,v\}` are the receive and transmit polarization, respectively (horizontal or vertical). Reciprocity is assumed, i.e. :math:`S_{hv} = S_{vh}`. Then the backscatter at a single pixel is fully represented by the complex target vector

.. math::

   \boldsymbol s = \begin{bmatrix} S_{hh} & S_{hv} & S_{vv} \end{bmatrix}^T


For multi-looked SAR data, backscatter values are averaged over :math:`n` pixels (to reduce speckle) and the backscatter may be represented appropriately by the (variance-)covariance matrix, which for fully polarimetric SAR data is given by

.. math::

   {\langle C \rangle}_\text{full} &= {\langle \boldsymbol s(i) \boldsymbol s(i)^H \rangle} =
   \begin{bmatrix}
   {\langle S_{hh}S_{hh}^* \rangle} & {\langle S_{hh}S_{hv}^* \rangle} & {\langle S_{hh}S_{vv}^* \rangle} \\
   {\langle S_{hv}S_{hh}^* \rangle} & {\langle S_{hv}S_{hv}^* \rangle} & {\langle S_{hv}S_{vv}^* \rangle} \\
   {\langle S_{vv}S_{hh}^* \rangle} & {\langle S_{vv}S_{hv}^* \rangle} & {\langle S_{vv}S_{vv}^* \rangle} \\
   \end{bmatrix}

where :math:`{\langle \cdot \rangle}` is the ensemble average, :math:`^*` denotes complex conjugation, and :math:`^H` is Hermitian conjugation.
Often, only one polarization is transmitted (e.g. horizontal), giving rise to dual polarimetric SAR data. In this case the covariance matrix is

.. math::

   {\langle C \rangle}_\text{dual} =
   \begin{bmatrix}
   {\langle S_{hh}S_{hh}^* \rangle} & {\langle S_{hh}S_{hv}^* \rangle} \\
   {\langle S_{hv}S_{hh}^* \rangle} & {\langle S_{hv}S_{hv}^* \rangle} \\
   \end{bmatrix}


These covariance matrices follow a complex Wishart distribution as follows:

.. math::

   \boldsymbol X_i \sim W_C(p,n,\boldsymbol\Sigma_i), \quad i = 1,...,k

where :math:`p` is the rank of :math:`\boldsymbol X_i = n {\langle \boldsymbol C_i \rangle}`, :math:`E[\boldsymbol X_i] = n \boldsymbol\Sigma_i`, and :math:`\boldsymbol\Sigma_i` is the expected value of the covariance matrix.

In the first instance, the change detection problem then becomes a test of the null hypothesis
:math:`H_0 : \boldsymbol\Sigma_1 = \boldsymbol\Sigma_2 = ... = \boldsymbol\Sigma_k`, i.e. whether the expected value of the backscatter remains constant. This test is a so-called omnibus test.

A test statistic for the omnibus test can be derived as:

.. math::

   Q = k^{pnk} \frac{\prod_{i=1}^k \left| \boldsymbol X_i \right|^n }{\left| \boldsymbol X \right|^{nk}}
   = \left\{ k^{pk} \frac{\prod_{i=1}^k \left| \boldsymbol X_i \right| }{\left| \boldsymbol X \right|^k} \right\}^n

where :math:`\boldsymbol X = \sum_{i=1}^k \boldsymbol X_i \sim W_C(p,nk,\boldsymbol\Sigma)`.
The test statistic can be translated into a probability :math:`p(H_0)`.
The hypothesis test is repeated iteratively over subsets of the time series in order to determine the actual time of change.


.. topic:: See Also:

 * :class:`nd.change.OmnibusTest`


.. topic:: References:

 * Conradsen, K., Nielsen, A. A., & Skriver, H. (2016).
   `Determining the Points of Change in Time Series of Polarimetric SAR Data <https://doi.org/10.1109/TGRS.2015.2510160>`_.
   IEEE Transactions on Geoscience and Remote Sensing, 54(5), 3007–3024.
