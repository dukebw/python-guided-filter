import numpy as np
import scipy as sp
import scipy.ndimage
import torch
import torch.nn.functional as F


def box(img, r):
    """ O(1) box filter
        img - >= 2d image
        r   - radius of box filter
    """
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)


    tile = [1] * img.ndim
    tile[0] = r
    imCum = np.cumsum(img, 0)
    imDst[0:r+1, :, ...] = imCum[r:2*r+1, :, ...]
    imDst[r+1:rows-r, :, ...] = imCum[2*r+1:rows, :, ...] - imCum[0:rows-2*r-1, :, ...]
    imDst[rows-r:rows, :, ...] = np.tile(imCum[rows-1:rows, :, ...], tile) - imCum[rows-2*r-1:rows-r-1, :, ...]

    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1, ...] = imCum[:, r:2*r+1, ...]
    imDst[:, r+1:cols-r, ...] = imCum[:, 2*r+1 : cols, ...] - imCum[:, 0 : cols-2*r-1, ...]
    imDst[:, cols-r: cols, ...] = np.tile(imCum[:, cols-1:cols, ...], tile) - imCum[:, cols-2*r-1 : cols-r-1, ...]

    return imDst

def _gf_color(I, p, r, eps, s=None):
    """ Color guided filter
    I - guide image (rgb)
    p - filtering input (single channel)
    r - window radius
    eps - regularization (roughly, variance of non-edge noise)
    s - subsampling factor for fast guided filter
    """
    fullI = I
    fullP = p
    if s is not None:
        I = sp.ndimage.zoom(fullI, [1/s, 1/s, 1], order=1)
        p = sp.ndimage.zoom(fullP, [1/s, 1/s], order=1)
        r = round(r / s)

    h, w = p.shape[:2]
    N = box(np.ones((h, w)), r)

    mI_r = box(I[:,:,0], r) / N
    mI_g = box(I[:,:,1], r) / N
    mI_b = box(I[:,:,2], r) / N

    mP = box(p, r) / N

    # mean of I * p
    mIp_r = box(I[:,:,0]*p, r) / N
    mIp_g = box(I[:,:,1]*p, r) / N
    mIp_b = box(I[:,:,2]*p, r) / N

    # per-patch covariance of (I, p)
    covIp_r = mIp_r - mI_r * mP
    covIp_g = mIp_g - mI_g * mP
    covIp_b = mIp_b - mI_b * mP

    # symmetric covariance matrix of I in each patch:
    #       rr rg rb
    #       rg gg gb
    #       rb gb bb
    var_I_rr = box(I[:,:,0] * I[:,:,0], r) / N - mI_r * mI_r;
    var_I_rg = box(I[:,:,0] * I[:,:,1], r) / N - mI_r * mI_g;
    var_I_rb = box(I[:,:,0] * I[:,:,2], r) / N - mI_r * mI_b;

    var_I_gg = box(I[:,:,1] * I[:,:,1], r) / N - mI_g * mI_g;
    var_I_gb = box(I[:,:,1] * I[:,:,2], r) / N - mI_g * mI_b;

    var_I_bb = box(I[:,:,2] * I[:,:,2], r) / N - mI_b * mI_b;

    a = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            sig = np.array([
                [var_I_rr[i,j], var_I_rg[i,j], var_I_rb[i,j]],
                [var_I_rg[i,j], var_I_gg[i,j], var_I_gb[i,j]],
                [var_I_rb[i,j], var_I_gb[i,j], var_I_bb[i,j]]
            ])
            covIp = np.array([covIp_r[i,j], covIp_g[i,j], covIp_b[i,j]])
            a[i,j,:] = np.linalg.solve(sig + eps * np.eye(3), covIp)

    b = mP - a[:,:,0] * mI_r - a[:,:,1] * mI_g - a[:,:,2] * mI_b

    meanA = box(a, r) / N[...,np.newaxis]
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, [s, s, 1], order=1)
        meanB = sp.ndimage.zoom(meanB, [s, s], order=1)

    q = np.sum(meanA * fullI, axis=2) + meanB

    return q


def _gf_gray(I, p, r, eps, s=None):
    """ grayscale (fast) guided filter
        I - guide image (1 channel)
        p - filter input (1 channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """
    I = I.astype(np.float64)
    p = p.astype(np.float64)
    if s is not None:
        Isub = sp.ndimage.zoom(I, 1/s, order=1)
        Psub = sp.ndimage.zoom(p, 1/s, order=1)
        r = round(r / s)
    else:
        Isub = I
        Psub = p

    (rows, cols) = Isub.shape

    # NOTE(brendan): torch
    k = 2*r + 1
    pad = (r, r, r, r)
    mean_kernel = torch.ones((1, 1, k, k), dtype=torch.float64)/k**2
    Isub_t = torch.from_numpy(Isub).unsqueeze(0).unsqueeze(0)
    meanI_t = F.conv2d(F.pad(Isub_t, pad=pad, mode='reflect'), mean_kernel)
    Psub_t = torch.from_numpy(Psub).unsqueeze(0).unsqueeze(0)
    meanP_t = F.conv2d(F.pad(Psub_t, pad=pad, mode='reflect'), mean_kernel)

    corrI_t = F.conv2d(F.pad(Isub_t*Isub_t, pad=pad, mode='reflect'), mean_kernel)
    corrIp_t = F.conv2d(F.pad(Isub_t*Psub_t, pad=pad, mode='reflect'), mean_kernel)
    varI_t = corrI_t - meanI_t**2
    covIp_t = corrIp_t - meanI_t*meanP_t

    a_t = covIp_t / (varI_t + eps)
    b_t = meanP_t - a_t * meanI_t

    meanA_t = F.conv2d(F.pad(a_t, pad=pad, mode='reflect'), mean_kernel)
    meanB_t = F.conv2d(F.pad(b_t, pad=pad, mode='reflect'), mean_kernel)

    if s is not None:
        meanA_t = F.interpolate(meanA_t, scale_factor=s, mode='bilinear')
        meanB_t = F.interpolate(meanB_t, scale_factor=s, mode='bilinear')

    I_t = torch.from_numpy(I).unsqueeze(0).unsqueeze(0)
    q_t = meanA_t * I_t + meanB_t

    # NOTE(brendan): numpy
    N = box(np.ones([rows, cols]), r)

    meanI = box(Isub, r) / N
    rows, cols = meanI.shape
    # NOTE(brendan): ignore border (messed up due to mirror).
    assert np.abs(meanI_t.squeeze()[r + 1:rows - r, r + 1:rows - r].numpy() - meanI[r + 1:rows - r, r + 1:rows - r]).max() < 1e-5
    meanP = box(Psub, r) / N
    assert np.abs(meanP_t.squeeze()[r + 1:rows - r, r + 1:rows - r].numpy() - meanP[r + 1:rows - r, r + 1:rows - r]).max() < 1e-5
    corrI = box(Isub * Isub, r) / N
    corrIp = box(Isub * Psub, r) / N
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP
    assert np.abs(corrI_t.squeeze()[r + 1:rows - r, r + 1:rows - r].numpy() - corrI[r + 1:rows - r, r + 1:rows - r]).max() < 1e-5
    assert np.abs(corrIp_t.squeeze()[r + 1:rows - r, r + 1:rows - r].numpy() - corrIp[r + 1:rows - r, r + 1:rows - r]).max() < 1e-5
    assert np.abs(varI_t.squeeze()[r + 1:rows - r, r + 1:rows - r].numpy() - varI[r + 1:rows - r, r + 1:rows - r]).max() < 1e-5
    assert np.abs(covIp_t.squeeze()[r + 1:rows - r, r + 1:rows - r].numpy() - covIp[r + 1:rows - r, r + 1:rows - r]).max() < 1e-5


    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = box(a, r) / N
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, s, order=1)
        meanB = sp.ndimage.zoom(meanB, s, order=1)

    q = meanA * I + meanB

    try:
        assert np.abs(a_t.squeeze()[r + 1:rows - r, r + 1:rows - r].numpy() - a[r + 1:rows - r, r + 1:rows - r]).max() < 1e-5
        assert np.abs(b_t.squeeze()[r + 1:rows - r, r + 1:rows - r].numpy() - b[r + 1:rows - r, r + 1:rows - r]).max() < 1e-5
        if s is None:
            r = 2*r
            assert np.abs(meanA_t.squeeze()[r + 1:rows - r, r + 1:rows - r].numpy() - meanA[r + 1:rows - r, r + 1:rows - r]).max() < 1e-5
            print('meanA ok')
            assert np.abs(meanB_t.squeeze()[r + 1:rows - r, r + 1:rows - r].numpy() - meanB[r + 1:rows - r, r + 1:rows - r]).max() < 1e-5
            print('meanB ok')
            assert np.abs(q_t.squeeze()[r + 1:rows - r, r + 1:rows - r].numpy() - q[r + 1:rows - r, r + 1:rows - r]).max() < 1e-5
            print('q ok')
    except AssertionError:
        import ipdb
        ipdb.set_trace()

    # return q
    return q_t.squeeze().numpy()


def _gf_colorgray(I, p, r, eps, s=None):
    """ automatically choose color or gray guided filter based on I's shape """
    if I.ndim == 2 or I.shape[2] == 1:
        return _gf_gray(I, p, r, eps, s)
    elif I.ndim == 3 and I.shape[2] == 3:
        return _gf_color(I, p, r, eps, s)
    else:
        print("Invalid guide dimensions:", I.shape)


def guided_filter(I, p, r, eps, s=None):
    """ run a guided filter per-channel on filtering input p
        I - guide image (1 or 3 channel)
        p - filter input (n channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """
    if p.ndim == 2:
        p3 = p[:,:,np.newaxis]

    out = np.zeros_like(p3)
    for ch in range(p3.shape[2]):
        out[:,:,ch] = _gf_colorgray(I, p3[:,:,ch], r, eps, s)
    return np.squeeze(out) if p.ndim == 2 else out


def test_gf():
    import imageio
    cat = imageio.imread('cat.bmp').astype(np.float32) / 255
    tulips = imageio.imread('tulips.bmp').astype(np.float32) / 255

    r = 8
    eps = 0.05

    cat_smoothed = guided_filter(cat, cat, r, eps)
    cat_smoothed_s4 = guided_filter(cat, cat, r, eps, s=4)

    imageio.imwrite('cat_smoothed.png', cat_smoothed)
    imageio.imwrite('cat_smoothed_s4.png', cat_smoothed_s4)

    tulips_smoothed4s = np.zeros_like(tulips)
    for i in range(3):
        tulips_smoothed4s[:,:,i] = guided_filter(tulips, tulips[:,:,i], r, eps, s=4)
    imageio.imwrite('tulips_smoothed4s.png', tulips_smoothed4s)

    tulips_smoothed = np.zeros_like(tulips)
    for i in range(3):
        tulips_smoothed[:,:,i] = guided_filter(tulips, tulips[:,:,i], r, eps)
    imageio.imwrite('tulips_smoothed.png', tulips_smoothed)


if __name__ == '__main__':
    with torch.no_grad():
        test_gf()
