#!/usr/bin/env python3

import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


class MakeWhiteout:

    def __init__(self, args):

        self.inputs = args.input
        self.outdir = Path(args.outdir)
        logging.debug("Using output directory %s" % self.outdir)

    def run(self):

        sz = len(self.inputs)
        logging.info("Processing %d images" % sz)

        for i in range(sz):
            infile = Path(self.inputs[i])
            outfile = self.outdir / infile.name
            logging.debug("Transforming %s to %s" % (infile, outfile))

            img = cv2.imread(str(infile))

            fimg = img.astype(float) / 256.0

            ## Convert to float

            exp = max(0.0, (sz-i-1)) / sz
            print(exp)

            fimg = np.power( fimg, exp )

            outimg = (fimg * 255).astype(np.ubyte)
            cv2.imwrite(str(outfile), outimg)


        return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', dest='outdir', nargs='?',
                        required=True,
                        help="Output directory (will be create if it doesn't exist)" )

    parser.add_argument("--log", nargs='?', default="INFO")

    parser.add_argument('input', nargs='+', help="Input images" )

    args = parser.parse_args()
    logging.getLogger().setLevel( args.log.upper() )

    if not os.path.exists( args.outdir ):
            os.makedirs( args.outdir )

    MakeWhiteout(args).run()
