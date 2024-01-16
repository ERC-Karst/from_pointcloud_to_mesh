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

    decimatedCloud = cc.loadPointCloud(os.path.join(root, p["paths"]["subsampledCut2DOutCloudName"]), cc.CC_SHIFT_MODE.XYZ, 0, *shift)

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
