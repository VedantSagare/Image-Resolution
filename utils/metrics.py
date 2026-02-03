from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(sr, hr):
    return peak_signal_noise_ratio(hr, sr, data_range=1.0)

def calculate_ssim(sr, hr):
    hr_img = hr.transpose(1, 2, 0)
    sr_img = sr.transpose(1, 2, 0)
    try:
        return structural_similarity(
            hr_img,
            sr_img,
            channel_axis=-1,
            data_range=1.0,
        )
    except TypeError:
        return structural_similarity(
            hr_img,
            sr_img,
            multichannel=True,
            data_range=1.0,
        )

