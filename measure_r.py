import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import cv2 as cv

import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
#import trackpy as tp


# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True, help = "Path to the image")
#args = vars(ap.parse_args())
#Read the images


#frames = pims.PyAVVideoReader("0.mp4")
img=cv.imread('0.png',cv.IMREAD_COLOR)
output=img.copy()


img = cv.medianBlur(img,5)
#cv.imshow('output',output)
#cv.waitKey(0)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,dp=1,minDist=1,minRadius=15,maxRadius=30)
print(circles)

# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
        cv.circle(output, (x, y), r, (0, 255, 0), 4)
        cv.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        cv.imshow("output",output)
        cv.waitKey(0)
# show the output image
'''
circles = np.uint16(np.aroound(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()
'''
