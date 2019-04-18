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

        if self.outputdir:
          outfile = open( self.outputdir / "detections.csv", "w")

        for i,imgname in enumerate(self.inputs):
            logging.debug("Detecting on %s" % imgname)

            ## Read first image
            img = cv2.imread(str(imgname))
            (kp,desc) = self.detectAndDescribe( img )

            self.detections[i] = {"kp": kp, "descriptors": desc }

            self.drawDetections( img, self.detections[i], output=self.workdir("features", imgname=imgname ) )

            logging.info("Detected %d features" % (len(self.detections[i]["kp"])))

            if outfile:
                outfile.write("%s\n" % ",".join([Path(self.inputs[i]).name, str(len(self.detections[i]["kp"]))]) )


    def detectAndDescribe(self, image):

        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image
        descriptor = cv2.ORB_create(nfeatures=2000)
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

            logging.debug("Checking %d %d %d" % (i,len(self.detections[i]["kp"]),len(self.detections[i+1]["kp"])))

            if not self.detections[i] or not self.detections[i+1]:
                continue

            if( len(self.detections[i]["kp"]) < 10 or len(self.detections[i+1]["kp"]) < 10 ):
              continue

            logging.info("Matches between %d and %d" % (i,i+1))

            self.matches[i] = self.match( self.detections[i], self.detections[i+1] )
            self.drawMatch( i,i+1 )


    def match( self, detA, detB ):

        matcher = cv2.BFMatcher( cv2.NORM_L2, crossCheck=True )

        matches = matcher.match( detA["descriptors"], detB["descriptors"])

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * 0.50)
        matches = matches[:numGoodMatches]

        return matches


    def drawMatch( self, i, j ):
        detA = self.detections[i]
        detB = self.detections[j]

        matches = self.matches[i]

        imgA = cv2.imread( str( self.inputs[i] ))
        imgB = cv2.imread( str( self.inputs[j] ))

        img = cv2.drawMatches( imgA, detA["kp"], imgB, detB["kp"], matches, None, flags=2 )

        cv2.imwrite( str( self.workdir("matches", imgname=self.inputs[i] )), img )


    ##=== Homography calculation ==

    def computeHomographyAll(self):

        self.homography = [None] * len(self.matches)

        if self.outputdir:
          outfile = open( self.outputdir / "homography.csv", "w")

        for i in range( 0, len(self.matches) ):

          if not(self.matches[i]):
            continue

          logging.debug("Computing homography %d" % i)

            ## Convert matches to points
          pointsA = np.zeros((len(self.matches[i]), 2), dtype=np.float32)
          pointsB = np.zeros((len(self.matches[i]), 2), dtype=np.float32)

          # logging.debug("Detection lengths: %d %d" % (len(self.detections[i  ]["kp"]),len(self.detections[i+1]["kp"])))
          # logging.debug("    Match lengths: %s %s" % (pointsA.shape, pointsB.shape))

          for j, match in enumerate(self.matches[i]):
            # logging.debug("%d Indices: %d %d" % (i, match.queryIdx, match.trainIdx))
            pointsA[j, :] = self.detections[i  ]["kp"][match.queryIdx].pt
            pointsB[j, :] = self.detections[i+1]["kp"][match.trainIdx].pt

          h,mask = cv2.findHomography( pointsA, pointsB, cv2.RANSAC )

          self.homography[i] = h

          self.drawHomography(i)

          if outfile:
              inliers = np.count_nonzero(mask)
              csv_out = ",".join([Path(self.inputs[i]).name, str(inliers)])
              outfile.write("%s\n" % csv_out )


        if outfile:
          outfile.close()


    def drawHomography(self,i):
        matches = self.homography[i]

        imgA = cv2.imread( str( self.inputs[i  ] ))
        imgB = cv2.imread( str( self.inputs[i+1] ))

        h,w,channels=imgA.shape

        imgBwarped = cv2.warpPerspective( imgB, np.linalg.inv(self.homography[i]), (w,h) )

        ## Overlay imgA onto imgB?

        alpha = 0.5
        imgBlended = alpha*imgBwarped + (1-alpha)*imgA

        cv2.imwrite( str( self.workdir("warped", imgname=self.inputs[i] )), imgBwarped )
        cv2.imwrite( str( self.workdir("blended", imgname=self.inputs[i] )), imgBlended )



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
