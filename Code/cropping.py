import cv2
import numpy as np
import sys
import math
from sklearn.cluster import DBSCAN


def getFeaturePoints(image):
	
	"""Takes as input an image, returns a matrix of coordinates of ORB feature points"""
	
	orb = cv2.ORB_create(nfeatures=100000)
	kp = orb.detect(image,None)
	kp, des = orb.compute(image, kp)
	
	matrixKp = np.zeros((len(kp),2))
	for i in range(len(kp)):
		matrixKp[i,0], matrixKp[i,1] = kp[i].pt
    
	return matrixKp


def createMatrixFromCoords(image, matrixKp):
	
	'''Takes as input an image and a matrix of coordinates of this image, 
	creates and returns a gray image of same width and height as original image, 
	with all pixels being black except for the feature points being white'''
	
	imageKp = np.zeros((image.shape[0],image.shape[1]))
	for i in range(matrixKp.shape[0]):
		imageKp[int(matrixKp[i,1]),int(matrixKp[i,0])]=255
        
	return imageKp.astype(np.uint8)


def cropCemetery(image):
	
	'''Takes as input an image, returns a cropped image with region of 
	maximum density of ORB feature points in the image '''
		
	featurePoints = getFeaturePoints(image)
	
	dbscan = DBSCAN(eps=20, min_samples=100)

	clustersFeaturePoints = dbscan.fit_predict(featurePoints)
	clusters, counts = np.unique(clustersFeaturePoints,      return_counts=True)

	densityMax=0
	for idx in np.argsort(counts)[-5:]:
	    nb = counts[idx]
	    featurePointsCluster = featurePoints[clustersFeaturePoints==clusters[idx]]
	    area = (np.max(featurePointsCluster[:,0])-np.min(featurePointsCluster[:,0]))*(np.max(featurePointsCluster[:,1])-np.min(featurePointsCluster[:,1]))
	    density = nb*nb/area
	    if density>densityMax:
	        idxMax=idx
	        densityMax=density
	        
	featurePointsBiggestCluster = featurePoints[clustersFeaturePoints==clusters[idxMax]]
	
	xMin = int(np.min(featurePointsBiggestCluster[:,0]))
	xMax = int(np.max(featurePointsBiggestCluster[:,0]))
	yMin = int(np.min(featurePointsBiggestCluster[:,1]))
	yMax = int(np.max(featurePointsBiggestCluster[:,1]))
	
	return image[yMin:yMax,xMin:xMax,:]