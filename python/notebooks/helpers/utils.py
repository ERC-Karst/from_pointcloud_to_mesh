import numpy as np 
import matplotlib.pyplot as plt
import os 

import cloudComPy as cc # cloud compare python interface.
if cc.isPluginCanupo():
    import cloudComPy.Canupo

if cc.isPluginPCL():
    import cloudComPy.PCL
from osgeo import gdal, ogr, osr
from subprocess import run

import json

from yaml import load
from yaml.loader import Loader

def plot_cloud_centreline(ax, pointset, centreline, view="PLAN", decim = 10):
    first_axis = 0
    third_axis = 2
    if view == "PLAN":
        second_axis = 1
        
    elif view == "PROFILE N":
        second_axis = 2
    elif view == "PROFILE E":
        second_axis = 2
        first_axis = 1
    else:
        print("View keyword not recognised.\n Use: 'PLAN', 'PROFILE N' or 'PROFILE E'.\nreverting to basic plan view.")
        second_axis = 1

    ax.plot(centreline[:, first_axis], centreline[:, second_axis], color = "w", lw = 4)
    ax.plot(centreline[:, first_axis], centreline[:, second_axis], color = "C1", lw = 2, label = "centreline", marker = "o", ls = "--")

    ax.scatter(pointset[::decim, first_axis], pointset[::decim, second_axis], c=pointset[::decim, third_axis], alpha = 0.05, edgecolor=None, s = 1)
    ax.set_aspect("equal")

    ax.legend()

    return ax

def plot_segmentation_scheme(root, ax):
    PROCESSING_FILEPATH = os.path.join(root, "params.yaml")

    p = load(open(PROCESSING_FILEPATH), Loader)    
    # load the processing parameters
    f  = open(os.path.join(root, p["paths"]["cropboxes"]))
    data = json.load(f)
    shift = p['alignment']['globalShift']
    polylineZ = 2000
    
    # convert the polyline features to np.arrays
    length_of_features = [len(feature["geometry"]["coordinates"][0][0]) for feature in data["features"]]

    polylines = [np.zeros((n, 3)) for n in length_of_features]

    # create a list of polyline entities 
    cc_polylines = []
    
    for c,feature in enumerate(data["features"]):
        coords = np.array(feature["geometry"]["coordinates"][0][0])

        polylines[c][:,0:2] = coords[:,0:2]
        polylines[c][:,2] = polylineZ
        
    for c, polyline in enumerate(polylines): 
        # create cloud from this and add to polyline object-
        boxCloud = cc.ccPointCloud("sectionAlongAxis")
        polyline_shifted = polyline + np.array(shift)
        boxCloud.coordsFromNPArray_copy(polyline_shifted)

        boxLine = cc.ccPolyline(boxCloud)
        boxLine.addChild(boxCloud)
        boxLine.addPointIndex(0, boxCloud.size())
        boxLine.setClosed(True)
        boxLine.setGlobalShift(*shift)
        cc_polylines.append(boxLine)

    decimatedCloud = cc.loadPointCloud(os.path.join(root, p["paths"]["subsampledGeorefOutCloudName"]), cc.CC_SHIFT_MODE.XYZ, 0, *shift)

    for c, line in enumerate(cc_polylines):
        xi = line.getAssociatedCloud().toNpArray()[:,0]
        yi = line.getAssociatedCloud().toNpArray()[:,1]
        centroid = line.getAssociatedCloud().computeGravityCenter()
        ax.plot(xi - shift[0], yi - shift[1], color = "r")
        ax.text(centroid[0]- shift[0], centroid[1] - shift[1], c,  ha = "center", va = "center", fontsize = 8)
    
    ax.set_aspect("equal")
    ax.scatter(decimatedCloud.toNpArray()[:,0]- shift[0], decimatedCloud.toNpArray()[:,1]- shift[1], s = 0.05, color = "lightgrey")

    return ax, (cc_polylines, decimatedCloud)



def load_matrix_from_file(filepath):
    """
    loads a cloud compare transformation matrix text file and
    returns a string representation.

    ----------
    
    arguments:

        fp -> str: the filepath

    ----------
    
    returns :

        m_data_str -> str : the string representation of 4 x 4 matrix
    
    
    """
    with open(filepath) as f:
        matrix_data = " ".join(f.readlines())
        f.close()

    return matrix_data


def suggest_global_shift(targets):
    """
    calculates the centroid of the target coordinates and returns the nearest coordinate triplet
    rounded to the nearest 100 m

    ----------
    
    arguments:

        targets -> np.array: a numpy array with N target coordinates (N x 3 matrix)
    ----------
    
    returns :

        globalShift -> np.array : a numpy array containing the suggested X, Y and Z shifts. 
    
    """

    centroid = -np.mean(targets.T, axis = 1).reshape((-1,1))
    globalShift = np.round(centroid/100,0) * 100

    return globalShift  

def reorder_points(points, ind):
    """
    reorders the array of points from an unordered output of a mesh / plane intersection.

    ----------
    
    arguments:

        points -> np.array: a numpy array with N target coordinates (N x 3 matrix)
        ind -> int : the index of the starting point
    ----------
    
    returns :

        ordered_points -> np.array : a numpy array containing the reordered data (N x 3 matrix)
    
    """
    points = list(points)
    points_new = [ points.pop(ind) ]  # initialize a new list of points with the known first point
    pcurr      = points_new[-1]       # initialize the current point (as the known point)
    k = 0
    while len(points)>0 and k <= 1e4:
        d      = np.linalg.norm(np.array(points) - np.array(pcurr), axis=1)  # distances between pcurr and all other remaining points
        ind    = d.argmin()                   # index of the closest point
        points_new.append( points.pop(ind) )  # append the closest point to points_new
        pcurr  = points_new[-1]               # update the current point
        k+=1
    if k == 1e4:
        print("max iterations reached")

    ordered_points = np.array(points_new)

    return ordered_points