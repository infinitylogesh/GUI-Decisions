"""Images: opening examples, capping their size, and encoding screenshots for the server."""

import base64, io, math, os

from PIL import Image

# max pixels per image for training and serving (0: as given). GUI-Owl uses 1,048,576 so training and
# serving see the same image; Gemma 4 resizes to its own fixed token budget either way.
IMAGE_CAP = int(os.environ.get("IMAGE_CAP", 0))


def cap_image(im, cap=None, factor=32):
    """Downscale to at most `cap` pixels, sides multiples of `factor` (Qwen3-VL: patch 16 x merge 2),
    so the processor keeps the size."""
    cap = IMAGE_CAP if cap is None else cap
    W, H = im.size
    if not cap or W * H <= cap:
        return im
    s = math.sqrt(cap / (W * H))
    w, h = max(factor, int(W * s) // factor * factor), max(factor, int(H * s) // factor * factor)
    return im.resize((w, h), Image.BICUBIC)


def image_of(ex):
    """An example's image as RGB; a path is opened only when needed (keeps thousands of examples cheap)."""
    im = ex["image"]
    if isinstance(im, str):
        im = Image.open(im)
    return cap_image(im.convert("RGB") if im.mode != "RGB" else im)


def image_size(ex):
    if "size" in ex:
        return ex["size"]
    im = ex["image"]
    if isinstance(im, str):
        with Image.open(im) as f:
            return f.size
    return im.size


# How screenshots travel to the server. The fast path, and the default for serving: JPEG at the model's
# input size (Gemma 4's encoder sees about 896 x 896 at its default budget): ~8 ms to build instead of
# ~118 ms for a full-size PNG, and cheaper for the server to decode. IMG_FMT=png IMG_MAX_PX=0 sends the
# lossless full-size image instead.
IMG_FMT = os.environ.get("IMG_FMT", "jpeg")
IMG_MAX_PX = int(os.environ.get("IMG_MAX_PX", 896 * 896))


def data_url(im, fmt=None, max_px=None):
    fmt = fmt or IMG_FMT
    max_px = IMG_MAX_PX if max_px is None else max_px
    im = im.convert("RGB")
    if max_px and im.width * im.height > max_px:
        s = (max_px / (im.width * im.height)) ** 0.5
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
    b = io.BytesIO()
    if fmt == "jpeg":
        im.save(b, "JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
