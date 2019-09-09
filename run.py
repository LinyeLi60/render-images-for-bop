"""
Script to generate train images for ssd-6d.
Usage:
    run.py [options]
    run.py (-h | --help)

Options:
    -d, --dataset=<string>   Path to SIXD dataset [default: tyol]
    -s, --sequence=<string>     Number of the sequence [default: 1 2]
    -h --help                Show this message and exit
"""

from docopt import docopt
import numpy as np
from rendering.utils import precompute_projections
import glob
from utils.sixd import load_sixd
import os
import math


def calc_azimuth(x, y):
    two_pi = 2.0 * math.pi
    return (math.atan2(y, x) + two_pi) % two_pi


args = docopt(__doc__)
sixd_base = 'dataset/' + args["--dataset"]
sequences = args["--sequence"].split()
pts = np.load("numpy_files/viewpoints.npy")

sequences = range(1, 21)
models = ['obj_{:06d}'.format(int(x)) for x in sequences]  # Overwrite model name
bench = load_sixd(sixd_base, sequences=sequences)
print('Models:', models)
print('Precomputing projections for each used model...')
# image_files = glob.glob(os.path.join(os.path.expanduser("~"), 'data', "images/*.jpg"))    # put coco images as background in this folder
model_map = bench.models  # Mapping from name to model3D instance
precompute_projections(args["--dataset"], pts, bench.cam, model_map, models)

