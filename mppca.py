"""
Run MP-PCA.
"""
import numpy as np


def denoise_cv(img, window=(5, 5, 5), mask=None):
    """
    Denoising implementation by Jonas Olesen, Mark Does and Sune Jespersen for diffusion
    MRI data based on the algorithm presented by Veraart et al. (2016) 142, p
    394-406 https://doi.org/10.1016/j.neuroimage.2016.08.016.

    Modified to remove mean across voxels (compute principal components of
    the covariance not correlation matrix).

    Parameters
    ----------
    image
        contains MRI image data. The first dimensions must
        discriminate between pixels, while the last dimension should correspond
        to b-value/ gradient variation. Thus image data could for instance be
        structured as [X,Y,Z,N] or [X,Y,N].
    window
        specifies the dimensions of the sliding window. For image data
        structured as [X,Y,Z,N], a window of [5 5 5] is typical.
    mask
        logical array specifying which pixels to include -- if image is
        [X,Y,Z,N] then mask is [X,Y,Z]. Can optionally be left unspecified in
        which case every pixel is included.

    Returns
    -------
    denoisedImage
        contains denoised image with same structure as input.
    S2
        contains estimate of variance in each pixel.
    P
        specifies the found number of principal components.

    Notes
    -----
    Free to use, but please cite Veraart et al. (2016) 142, p
    394-406 https://doi.org/10.1016/j.neuroimage.2016.08.016 and Does et al.,
    MRM (2018) (Evaluation of Principal Component Analysis Image Denoising on
    Multi-Exponential MRI Relaxometry), reference will be finalized when
    available.
    """
    dims = img.shape
    assert len(window) > 1 and len(window) < 4
    assert all(window > 0)
    assert all([window[i] < dims[i] for i in range(len(window))])
    if len(window) == 2:
        window = window + (1,)

    # denoise image
    n_vols = dims[3]
    window_size = np.prod(window)
    denoised = np.zeros(dims)
    P = np.zeros(dims[:3])
    S2 = np.zeros(P.shape)
    counter = np.zeros(P.shape)
    m = dims[0] - window[0] + 1
    n = dims[1] - window[1] + 1
    o = dims[2] - window[2] + 1

    for index in range(np.prod([m, n, o])):
        # TODO: Clarify order of operations with parens
        i = index - (k - 1) * m * n - (j - 1) * m  # not set for zero-indexing
        j = np.floor((index - (k - 1) * m * n) / m) + 1  # should be fine for zero-indexing
        k = np.floor(index / m / n) + 1  # should be fine for zero-indexing

        rows = np.arange(i, i + window[0])
        cols = np.arange(j, j + window[1])
        slis = np.arange(k, k + window[2])

        # Check mask
        # NOTE: I think this is the searchlight mask
        mask_check = np.reshape(mask[rows, cols, slis], [window_size, 1]).T
        if np.all(~mask_check):
            continue

        # Create X data matrix
        # NOTE: This is the searchlight data
        X = np.reshape(img.get_data()[rows, cols, slis, :], [window_size, n_vols]).T

        # Remove voxels not contained in mask
        X[:, ~mask_check] = []  # ??
        if X.shape[1] == 1:
            continue

        new_X = np.zeros((n_vols, window_size))
        sigma2 = np.zeros((1, window_size))
        p = np.zeros((1, window_size))
        temp_new_X, temp_sigma2, temp_p = denoise_matrix(X)
        new_X[:, mask_check] = temp_new_X
        sigma2[mask_check] = temp_sigma2
        p[mask_check] = temp_p

        # Assign new_X to correct indices in denoised_img
        denoised[rows, cols, slis, :] = denoised[rows, cols, slis, :] + np.reshape(new_X.T, (window, n_vols))
        P[rows, cols, slis] = P[rows, cols, slis] + np.reshape(p, window)
        S2[rows, cols, slis] = S2[rows, cols, slis] + np.reshape(sigma2, window)
        counter[rows, cols, slis] += 1

    skip_check = mask and counter == 0
    counter[counter == 0] = 1
    # TODO: Translate to Python
    denoised = bsxfun(@rdivide, denoised, counter)
    P = bsxfun(@rdivide, P, counter)
    S2 = bsxfun(@rdivide, S2, counter)

    # adjust output to match input dimensions
    # assign original data to denoised_img outside of mask and at skipped voxels
    # TODO: Translate to Python
    original = bsxfun(@times, image, ~mask)
    denoised = denoised + original
    original = bsxfun(@times, image, skipCheck)
    denoised = denoised + original

    # Shape denoisedImage as original image
    if len(dimsOld) == 3:
        denoised = np.reshape(denoised, dimsOld)
        S2 = np.reshape(S2, dimsOld[:-1])
        P = np.reshape(P, dimsOld[:-1])

    return denoised, S2, P


def denoise_matrix(X):
    """
    helper function to denoise.m
    Takes as input matrix X with dimension MxN with window_size corresponding to the
    number of pixels and n_vols to the number of data points. The output consists
    of "newX" containing a denoised version of X, "sigma2" an approximation
    to the data variation, "p" the number of signal carrying components.
    """
    n_vols, window_size = X.shape
    min_mn = np.min(X.shape)
    X_m = np.mean(X, axis=1)  # MDD added Jan 2018; mean added back to signal below
    X = X - X_m
    # [U,S,V] = svdecon(X); MDD replaced with MATLAB svd vvv 3Nov2017
    # U, S, V = svd(X, 'econ')
    # NOTE: full matrices=False should be same as economy-size SVD
    U, S, V = np.linalg.svd(X, full_matrices=False)

    lambda_ = (np.diag(S) ** 2) / window_size

    p = 0
    p_test = False
    scaling = (n_vols - np.arange(0, min_mn)) / window_size
    scaling[scaling < 1] = 1
    while not p_test:
        sigma2 = (lambda_[p + 1] - lambda_[min_mn]) / (4 * np.sqrt((n_vols - p) / window_size))
        p_test = np.sum(lambda_[p + 1:min_mn]) / scaling[p + 1] >= (min_mn - p) * sigma2
        if not p_test:
            p += 1

    sigma2 = np.sum(lambda_[p + 1:min_mn]) / (min_mn - p) / scaling[p + 1]

    new_X = np.dot(np.dot(U[:, 1:p], S[1:p, 1:p]), V[:, 1:p].T) + X_m
    return new_X, sigma2, p
