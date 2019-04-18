#!/usr/bin/env python3

import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


class MakeProducts:

    def __init__(self, args):

        self.inputs = args.input
        self.outdir = Path(args.outdir)
        logging.debug("Using output directory %s" % self.outdir)

    def makeWhiteout(self):

        sz = len(self.inputs)
        logging.info("Processing %d images" % sz)

        for i in range(sz):
            infile = Path(self.inputs[i])
            outfile = self.outdir / infile.name
            logging.debug("Transforming %s to %s" % (infile, outfile))

            img = cv2.imread(str(infile))

            fimg = img.astype(float) / 255.0

            ## Convert to float
            exp = np.power(max(0.0, (sz-i-1)) / sz, 2)
            print(exp)

            fimg = np.power( fimg, exp )
            fimg = np.clip( fimg, np.power(0.02, exp), 255 )

            outimg = (fimg * 255).astype(np.ubyte)
            cv2.imwrite(str(outfile), outimg)

        return

    def makeSwir(self):

        for file in self.inputs:
            infile = Path(file)
            outfile = self.outdir / infile.name
            logging.debug("Transforming %s to %s" % (infile, outfile))

            img = cv2.imread(str(infile))

            img = cv2.resize( img, (640,480) )
            img = cv2.resize( img, (1024,768) )

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            fimg = img.astype(float) / 256.0
            fimg = 1-fimg

            outimg = (fimg * 255).astype(np.ubyte)
            cv2.imwrite(str(outfile), outimg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', dest='outdir', nargs='?',
                        required=True,
                        help="Output directory (will be create if it doesn't exist)" )

    parser.add_argument("--whiteout", action="store_true")

    parser.add_argument("--false-swir", dest='swir', action="store_true")

    parser.add_argument("--log", nargs='?', default="INFO")

    parser.add_argument('input', nargs='+', help="Input images" )

    args = parser.parse_args()
    logging.getLogger().setLevel( args.log.upper() )

    if not os.path.exists( args.outdir ):
            os.makedirs( args.outdir )


    make = MakeProducts(args)

    if args.whiteout:
        make.makeWhiteout()
    elif args.swir:
        make.makeSwir()
