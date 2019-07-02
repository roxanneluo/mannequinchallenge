#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
import argparse
import numpy as np
import os
from PIL import Image
from skimage.transform import resize
import struct
from subprocess import call
import warnings
import six

if six.PY2:
    class ResourceWarning(RuntimeWarning):
        pass

# Needed to suppress ResourceWarning for unclosed image file on dev server.
warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", UserWarning)


# resizes the image
def resize_to_target(image, max_size):
    print("original size: %d x %d" % (image.shape[1], image.shape[0]))

    scale = max_size / float(max(image.shape[0], image.shape[1]))
    resized_height = int(image.shape[0] * scale)
    resized_width = int(image.shape[1] * scale)
    if resized_height % 16 != 0:
        resized_height = resized_height - (resized_height % 16)
        print("Warning clipping height to closest multiple of 16.")
    if resized_width % 16 != 0:
        resized_width = resized_width - (resized_width % 16)
        print("Warning clipping width to closest multiple of 16.")

    print("resized: %d x %d" % (resized_width, resized_height))
    image = resize(image, (resized_height, resized_width), order=1, mode="reflect")

    return image


# Reads an image and returns a normalized float buffer (0-1 range). Corrects
# rotation based on EXIF tags.
def load_image(file_name, max_size=None):
    img, angle = load_image_angle(file_name, max_size)
    return img


def load_image_angle(file_name, max_size=None, min_size=None, angle=0):

    with Image.open(file_name) as img:

        if hasattr(img, "_getexif") and img._getexif() is not None:
            # orientation tag in EXIF data is 274
            exif = dict(img._getexif().items())

            # adjust the rotation
            if 274 in exif:
                if exif[274] == 8:
                    angle = 90
                elif exif[274] == 6:
                    angle = 270
                elif exif[274] == 3:
                    angle = 180

        if angle != 0:
            img = img.rotate(angle, expand=True)

        img = np.float32(img) / 255.0

        if max_size is not None:
            if min_size is not None:
                img = resize(img, (max_size, min_size), order=1, mode="reflect")
            else:
                img = resize_to_target(img, max_size)

        return img, angle

    return [[]], 0.0


# Save image to binary file, so that it can be read in C++ with
# #include "compphotolib/core/CvUtil.h"
# freadimg(fileName, image);
def save_raw_float32_image(file_name, image):
    with open(file_name, "wb") as f:
        CV_CN_MAX = 512
        CV_CN_SHIFT = 3
        CV_32F = 5

        dims = image.shape
        h = 0
        w = 0
        d = 1
        if len(dims) == 2:
            h, w = image.shape
            float32_image = np.transpose(image).astype(np.float32)
        else:
            h, w, d = image.shape
            float32_image = np.transpose(image, [2, 1, 0]).astype("float32")

        cv_type = CV_32F + ((d - 1) << CV_CN_SHIFT)

        pixel_size = d * 4

        if d > CV_CN_MAX:
            raise Exception("Cannot save image with more than 512 channels")
        f.write(struct.pack("i", h))
        f.write(struct.pack("i", w))
        f.write(struct.pack("i", cv_type))
        f.write(struct.pack("Q", pixel_size))  # Write size_t ~ uint64_t

        # Set buffer size to 16 MiB to hide the Python loop overhead.
        buffersize = max(16 * 1024 ** 2 // image.itemsize, 1)

        for chunk in np.nditer(
            float32_image,
            flags=["external_loop", "buffered", "zerosize_ok"],
            buffersize=buffersize,
            order="F",
        ):
            f.write(chunk.tobytes("C"))


def save_image(file_name, image):
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".raw":
        save_raw_float32_image(file_name, image)
    else:
        image = 255.0 * image
        image = Image.fromarray(image.astype("uint8"))
        image.save(file_name)


def save_depth_map_colored(file_name, depth_map, color_binary):
    save_image(file_name, depth_map)
    color_depth_name = os.path.splitext(file_name)[0] + "_color.jpg"
    if color_binary != "":
        call([color_binary, "--inputFile", file_name, "--outputFile", color_depth_name])


# main print_function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, help="input image")
    parser.add_argument("--output_image", type=str, help="output image")
    parser.add_argument(
        "--max_size", type=int, default=768, help="max size of long image dimension"
    )
    args, unknown = parser.parse_known_args()

    img = load_image(args.input_image, int(args.max_size))
    save_image(args.output_image, img)
