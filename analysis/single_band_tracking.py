#!/usr/bin/env python3

import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from random import randrange


class SingleBandTracker:

    def __init__(self, args):

        self.file = args.input



    def run():


        return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?' )

    parser.add_argument("log", nargs='+', defatul="INFO")

    args = parser.parse_args()
    logger.setLevel( args.log.to_upper() )

    tracker = SingleBandTracker(args)

    tracker.run()
