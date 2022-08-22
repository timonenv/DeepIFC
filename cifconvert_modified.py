"""
Modified cifconvert from Lippeveld et al. 
https://github.com/saeyslab/cifconvert/blob/master/cifconvert/main.py

example run: python /path/to/cifconvert/main.py *cif filename* *output filename* *width* *height* --channels *e.g. 1 2 8 9 11 12* 
"""

import math
import javabridge
import bioformats as bf
import numpy as np
import multiprocessing
import logging
from tqdm import tqdm
from os.path import basename, splitext, join, exists
import argparse
import h5py

LOGGER = logging.getLogger(__name__)

def _pad_or_crop(image, _w, _h):

    pad_x = float(max(image.shape[1], _w) - image.shape[1])
    pad_y = float(max(image.shape[0], _h) - image.shape[0])

    pad_width_x = (int(math.floor(pad_x / 2)), int(math.ceil(pad_x / 2)))
    pad_width_y = (int(math.floor(pad_y / 2)), int(math.ceil(pad_y / 2)))
    def normal(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
        return vector

    if (_h > image.shape[0]) and (_w > image.shape[1]):
        return np.pad(image, (pad_width_y, pad_width_x), normal)
    else:
        if _w > image.shape[1]:
            temp_image = np.pad(image, pad_width_x, normal)
        else:
            if _h > image.shape[0]:
                temp_image = np.pad(image, pad_width_y, normal)
            else:
                temp_image = image

        return temp_image[
               int((temp_image.shape[0] - _h)/2):int((temp_image.shape[0] + _h)/2),
               int((temp_image.shape[1] - _w)/2):int((temp_image.shape[1] + _w)/2)
               ]


def _process_partial_cif(_id, _cif, _w, _h, _cpu_count, output, _c=None):

    logger = logging.getLogger(__name__)

    try:
        logger.debug("Starting Java VM in thread %s" % str(_id))
        javabridge.start_vm(class_path=bf.JARS, run_headless=True, max_heap_size="8G")
        logger.debug("Started Java VM in thread %s" % str(_id))

        fn = join(output, "%s_%s.tar.gz" % (splitext(basename(_cif))[0], str(_id)))
        logger.debug("Thread %s writing to %s" % (str(_id), fn))

        reader = bf.formatreader.get_image_reader("tmp", path=_cif)

        r_length = javabridge.call(reader.metadata, "getImageCount", "()I")
        num_channels = javabridge.call(reader.metadata, "getChannelCount", "(I)I", 0)
        print("Image count: {}. Channel count: {}.".format(r_length, num_channels))
        
        if _c is None:
            crange = [i for i in range(num_channels)]
        else:
            crange = np.array(_c)-1

        with h5py.File(output, "a") as f:
            images = []
            masks = []
            for c in crange:
                channelgrp = f.create_group("channel_%d" % (c+1))
                images.append(channelgrp.create_dataset("images", shape=(int(r_length/2), _w, _h), dtype="<u2"))
                masks.append(channelgrp.create_dataset("masks", shape=(int(r_length/2), _w, _h), dtype="?"))

            print(
                "Thread %s extracting images %s to %s" % (
                    str(_id), str(_id*r_length), str((_id+1)*r_length)))

            for c_idx, c in enumerate(crange):
                print("Channel %d" % (c+1))
                for i in tqdm(range(_id*r_length, (_id+1)*r_length)[::2]): #step oli 2
                    images[c_idx][int(i/2)] = _pad_or_crop(reader.read(c=c, series=i, rescale=False), _w, _h)
                    masks[c_idx][int(i/2)] = np.array(_pad_or_crop(reader.read(c=c, series=i+1, rescale=False), _w, _h), dtype=bool)
    finally:
        javabridge.kill_vm()


def init(cif, output, width, height, channels, debug):

    if debug:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)

    if not exists(cif):
        LOGGER.error("cif input cannot be found (%s)." % cif)
        return

    if exists(output):
        LOGGER.error("Output file %s exists: appending to existing" % output)

    cpu_count = int(multiprocessing.cpu_count()/2)
    print("cpu count ", cpu_count)

    LOGGER.info("Starting %s jobs to extract images" % str(cpu_count))
    _process_partial_cif(
            0, cif,
            width, height, cpu_count,
            output, _c=channels
    )

def main():
    parser = argparse.ArgumentParser(prog="cifconvert")
    parser.add_argument(
        "cif", type=str,
        help="Input cif containing images.")
    parser.add_argument(
        "output", type=str,
        help="Output filename for h5."
    )
    parser.add_argument(
        "width", type=int,
        help="Width to crop or pad images to.")
    parser.add_argument(
        "height", type=int,
        help="Height to crop or pad images to.")
    parser.add_argument(
        "--channels", "-c", nargs="*", type=int, default=None,
        help="Images from these channels will be extracted. Default is to extract all.")
    parser.add_argument(
        "--debug", help="Show debugging information.", action='store_true'
    )

    flags = parser.parse_args()

    init(**vars(flags))


if __name__ == "__main__":
    main()
