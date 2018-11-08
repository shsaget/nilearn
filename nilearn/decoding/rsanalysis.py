""" 
Module for Representational Similarity Analysis RSA
This approach consists in observing the similarity between various 
conditions in their representation's space.
Overall this approach alows to compare these simili represeantions between 
various models (like the primate brains, computationnal models, behavioural models...)

Functions in this module are specially implemented for using neuro-imaging fMRI data.
Their inputs are the extracted signal values from brain images. 
Please see ****_values_extraction.py


"""
# Authors : Shinji Saget (sh.saget.pro@gmail.com)
#
#
#
#

import sys
import numpy as np
import pandas as pd
import 

######################################
## fMRI Signal Extraction Functions ##
######################################

def roi_values_extraction(........):
  """
  Extract signal values from .nii images using masks defining Regions Of Interest 
  """
  
def searchlight_values_extraction(..............):
  """
  Extract signal values from .nii images using a main mask defining Regions Of 
  Interest and several spherical masks centered in each voxel contained the main ROI
  """

############################################################
## First Level RSAnalysis (computing from a single model) ##
############################################################

def 

#    get_basic_DSM : permet de calculer une matrice de dissimilarité grâce à différente représentations
#                     exemple 25 vecteurs de longueurs identique correspondant à des condition différentes
#    get

#################################################
## Second Level RSAnalysis (models comparisons ##
#################################################
