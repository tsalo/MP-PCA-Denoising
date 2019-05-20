"""
Run MP-PCA.
"""
import numpy as np
import nibabel as nib
from nilearn.masking import apply_mask, unmask


def mppca_denoise(img, window=(5, 5, 5), mask=None):
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
    if isinstance(img, str):
        img = nib.load(img)

    dims = img.shape
    assert len(window) > 1 and len(window) < 4
    assert all(np.array(window) > 0)
    assert all([window[i] < dims[i] for i in range(len(window))])

    # Preallocate arrays
    denoised = img.get_data()
    S2 = np.zeros(img.shape[:3])
    P = np.zeros(img.shape[:3])
    counter = np.zeros(img.shape[:3], int)

    # Load mask
    if isinstance(mask, str):
        mask = nib.load(mask)

    if mask:
        mask_data = mask.get_data()
    else:
        mask_data = np.ones(img.shape)

    mask_idx = np.vstack(np.where(mask_data))
    for i in range(mask_idx.shape[1]):
        i, j, k = mask_idx[:, i]

        # Define 3D window
        i_min = np.max((i - window[0], 0))
        i_max = np.min((i + window[0], mask_data.shape[0]))
        j_min = np.max((j - window[1], 0))
        j_max = np.min((j + window[1], mask_data.shape[1]))
        k_min = np.max((k - window[2], 0))
        k_max = np.min((k + window[2], mask_data.shape[2]))
        window_mask = mask_data[i_min:i_max, j_min:j_max, k_min:k_max]
        # skip if only one voxel of window in mask
        if window_mask.sum() < 2:
            continue

        temp_mask = np.zeros(mask_data.shape, int)
        temp_mask[i_min:i_max, j_min:j_max, k_min:k_max] = window_mask
        temp_mask_img = nib.Nifti1Image(temp_mask, img.affine)

        masked_data = apply_mask(img, temp_mask_img).T  # (n_voxels, n_vols)
        denoised_data, sigma2, p = denoise_matrix(masked_data)

        # Convert scalars to window-sized arrays
        sigma2 = sigma2 * np.ones(denoised_data.shape[0])
        p = p * np.ones(denoised_data.shape[0])

        # create full-sized zero arrays with only window filled with real values
        unmasked_data = unmask(denoised_data.T, temp_mask_img).get_data()
        unmasked_sigma2 = unmask(sigma2, temp_mask_img).get_data()
        unmasked_p = unmask(p, temp_mask_img).get_data()

        # add denoised values to arrays
        denoised += unmasked_data
        S2 += unmasked_sigma2
        P += unmasked_p
        counter += temp_mask_img.get_data()

    # Divide the summed arrays by the number of times each voxel
    # is used across windows to get an average
    temp_counter = counter.copy()
    temp_counter[temp_counter == 0] = 1  # workaround for divide-by-zero errors
    denoised = denoised / temp_counter[..., None]
    S2 = S2 / temp_counter
    P = P / temp_counter

    # Fill denoised data not in mask or skipped by loop
    # with original data
    inv_mask = (1 - mask_data).astype(bool)
    skipped_voxels = (mask_data & ~counter).astype(bool)
    denoised[inv_mask, :] = original[inv_mask, :]
    denoised[skipped_voxels, :] = original[skipped_voxels, :]

    # Make imgs
    denoised = nib.Nifti1Image(denoised, img.affine)
    S2 = nib.Nifti1Image(S2, img.affine)
    P = nib.Nifti1Image(P, img.affine)

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
    X_m = np.mean(X, axis=1, keepdims=True)  # MDD added Jan 2018; mean added back to signal below
    X = X - X_m
    # [U,S,V] = svdecon(X); MDD replaced with MATLAB svd vvv 3Nov2017
    # U, S, V = svd(X, 'econ')
    # NOTE: full matrices=False should be same as economy-size SVD
    U, S, V = np.linalg.svd(X, full_matrices=False)
    S = np.diag(S)  # make S array into diagonal matrix

    lambda_ = (np.diag(S) ** 2) / window_size

    p = 0
    p_test = False
    scaling = (n_vols - np.arange(0, min_mn)) / window_size
    scaling[scaling < 1] = 1
    while not p_test:
        sigma2 = (lambda_[p] - lambda_[min_mn - 1]) / (4 * np.sqrt((n_vols - p) / window_size))
        p_test = np.sum(lambda_[p:min_mn-1]) / scaling[p] >= (min_mn - p - 1) * sigma2
        if not p_test:
            p += 1

    sigma2 = np.sum(lambda_[p:min_mn-1]) / (min_mn - p - 1) / scaling[p]
    new_X = np.dot(np.dot(U[:, :p], S[:p, :p]), V[:, :p].T) + X_m
    return new_X, sigma2, p
