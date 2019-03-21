#!/usr/bin/env python3

import cv2
import numpy as np
import logging
import argparse
import os
import matplotlib.pyplot as plt
from random import randrange
from pathlib import Path


class SingleBandTracker:

    def __init__(self, args):

        self.inputs = args.input
        self.outputdir = Path(args.outdir)

    def workdir( self, name, imgname = None ):
        d = self.outputdir / name

        if not d.is_dir():
            d.mkdir(parents=True)

        if imgname:
            d = d / Path(imgname).name

        return d


    def run(self):

        sz = len(self.inputs)

        self.detectAll()

        self.matchAll()

        self.computeHomographyAll()




        return


    ##=== Detection and description ====

    def detectAll( self ):
        self.detections = [None]*len(self.inputs)

        for i,imgname in enumerate(self.inputs):
            logging.debug("Detecting on %s" % imgname)

            ## Read first image
            img = cv2.imread(str(imgname))
            (kp,desc) = self.detectAndDescribe( img )

            self.detections[i] = {"kp": kp, "descriptors": desc }

            self.drawDetections( img, self.detections[i], output=self.workdir("SURF", imgname=imgname ) )

            logging.info("Detected %d features" % (len(self.detections[i]["kp"])))


    def detectAndDescribe(self, image):

        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image
        descriptor = cv2.ORB_create(nfeatures=100)
        (kps, features) = descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        #kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def drawDetections(self, img, detections, output=None ):

        kpimg = cv2.drawKeypoints( img, detections["kp"], None, color=(0,255,0), flags=0)
        print(output)
        cv2.imwrite( str(output), kpimg )


    ##== Matching ===

    def matchAll(self):
        self.matches = [None] * (len(self.detections)-1)

        for i in range( 0, len(self.detections)-1 ):

            self.matches[i] = self.match( self.detections[i], self.detections[i+1] )
            self.drawMatch( i,i+1 )


    def match( self, detA, detB ):

        matcher = cv2.BFMatcher( cv2.NORM_L2, crossCheck=True )

        matches = matcher.match( detA["descriptors"], detB["descriptors"])

        return matches


    def drawMatch( self, i, j ):
        detA = self.detections[i]
        detB = self.detections[j]

        matches = self.matches[i]

        imgA = cv2.imread( str( self.inputs[i] ))
        imgB = cv2.imread( str( self.inputs[j] ))

        img = cv2.drawMatches( imgA, detA["kp"], imgB, detB["kp"], matches[:10], None, flags=2 )

        cv2.imwrite( str( self.workdir("matches", imgname=self.inputs[i] )), img )


    ##=== Homography calculation ==

    def computeHomographyAll(self):

        self.homography = [None] * len(self.matches)

        for i in range( 0, len(self.matches) ):

            ## Convert matches to points
            pointsA = [ self.detections[i  ]["kp"][j.queryIdx].pt for j in self.matches[i] ]
            pointsB = [ self.detections[i+1]["kp"][j.trainIdx].pt for j in self.matches[i] ]

            self.homography[i] = cv2.computeHomography( pointsA, pointsB, cv2.RANSAC )


            print(self.homography[i])





##==== =======

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


    make = SingleBandTracker(args)

    make.run()
