from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(sr, hr):
    return peak_signal_noise_ratio(hr, sr, data_range=1.0)

def calculate_ssim(sr, hr):
    return structural_similarity(
        hr.transpose(1,2,0),
        sr.transpose(1,2,0),
        multichannel=True,
        data_range=1.0
    )

