""" 
Module for Representational Similarity Analysis RSA  (Kriegeskorte 2008)
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
#############
## IMPORTS ##
#############

import sys
import numpy as np
import pandas as pd

#############################################################################################################################
## fMRI Signal Extraction Functions #####################################################  fMRI Signal Extraction Functions##
#############################################################################################################################

def roi_values_extraction(........):
  """
  Extract signal values from .nii images using masks defining Regions Of Interest 
  """
  
def searchlight_values_extraction(..............):
  """
  Extract signal values from .nii images using a main mask defining Regions Of 
  Interest and several spherical masks centered in each voxel contained the main ROI
  """

#############################################################################################################################
## Function Utilities for Treating the Data ##################################### Function Utilities for Treating the Data ##
#############################################################################################################################

# Whitening/demeaning...
  

#############################################################################################################################
## First Level RSAnalysis (computing from a single model) ######### First Level RSAnalysis (computing from a single model) ##
#############################################################################################################################

def get_dsm_from_arrays(............):
  """
  Usefull for computationnal model
  From several arrays of signal values corresponding to various condition,
  compute dissimilarity matrice of these condition.
  The dissimilarity can be computed with various metrics as euclidean distance or 
  spearman (pearson ranked) correlation.
  """
 
def get_dsm_from_h5(...........):
  """ PAS FORCEMENT TRES UTILE !
  From a H5 file containing signal of several 
  compute dissimilarity matrice of these condition.
  The dissimilarity can be computed with various metrics as euclidean distance or 
  spearman (pearson ranked) correlation.
  """

def compute_searchlight_and_get_dsm(.....):
  """ 
  Group 'searchlight_values_extraction()' and 'get_dsm()' functions.
  Usefull is the signal values have not been already extracted or if they don't interest user.
  Consider about storing this extracted signal if you have sufficient space instead 
  of recomputing in each call of this function
  """

#############################################################################################################################
## Second Level RSAnalysis (models comparisons) ############################# Second Level RSAnalysis (models comparisons) ##
#############################################################################################################################

def compare_2_models(DSM1,DSM2):
  """ 
  Compare the DSM created from to different representationnal space 
  (diffrent model of different values)
  Spearman correlation is generaly used for this comparison of 
  DSMs but other metrics are implemented
  """

def cand_dsms_to_one_ref(DSMs_candidate , DSM_reference):
  """ 
  The idea there is to take a "candidate" set of DSMs and a "reference" DSM.
  All candidates will be correlated to the reference and the most matching 
  candidate will be saved.
  This saving take the form of and one-hot encoded array with the winning DSM 
  candidate index.
  
  e.g. :
  7 candidates are compared to the reference. 
  A first array is computed with the score of each candidate : 
  [0.5, 0.1, 0.3, 0.9, 0.4, 0.4, 0.7] 
  The second array one hot encode the first using the hghest score : 
  [  0  , 0,   0,   1,   0,   0,   0]
  """
  
  
  
def cand_dsms_to_ref_dsms(DSMs_candidate , DSMs_reference , attribution_mode = 'auto'):
  """ 
  The idea there is to take 2 sets of DSMs. A "candidate" set and a "reference" one.
  All candidates will be correlated to all references and the maximum correspondances will be save.
  The finality of that is an attribution of the candidate dsms to the reference ones.
  
  This attribution take the form of an array 2xL with L dipendending on the 'attribution_mode' :
    - "on_reference" :  L is the reference set length. 
                        The first row of the array is the index of the ref. DSMs.
                        Second row is the index of the cand. DSM attribuated to each ref.
    - "on_candidate" :  L is the candidate set length. 
                        The first row of the array is the index of the cand. DSMs.
                        Second row is the index of the ref. DSM on which each cand has been attribuated.
    - "auto"         :  compare the length of the both sets and :
                        - act like "on_reference" mode if ref. set is the largest
                        - act like "on_candidate" mode if cand. set is the largest
  
  Take care about the attribution_mode : if L is taken from the smallest set, the second row will no more
  be composed by scalar but by arrays that will modify the output dtype!
  If you don't know or you are not sure about the length of the input sets, consider using 'auto' mode
  """

