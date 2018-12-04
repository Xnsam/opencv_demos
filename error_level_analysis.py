"""Error Level analysis in python."""

from PIL import Image, ImageChops, ImageEnhance
from logzero import logger


quality = 90


def ela(fname, save_dir):
    """Generate an ela image on save_dir."""
    tmp_name = 'imgs/demos/img_ela_tmp.jpg'
    im = Image.open(fname)
    im.save(tmp_name, 'JPEG', quality=quality)

    tmp_fname_im = Image.open(tmp_name)
    ela_im = ImageChops.difference(im, tmp_fname_im)

    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    ela_im.save(save_dir)
    # os.remove()


fname = 'imgs/comp1.jpg'
save_dir = 'imgs/demos/ela.png'

logger.info('starting error level analysis...')
ela(fname, save_dir)
logger.info('completed error level analysis...')
