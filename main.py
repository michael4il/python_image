import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math
%matplotlib inline
def image_pre_process(img): # get image return gardient direction for each pixel in range [0,1]
    dx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    dy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    # magnitude = np.sqrt(dx ** 2 + dy ** 2)
    dir = np.arctan(dy / (np.abs(dx) + 1) ) # divde by zero warning 
    dir = dir - np.min(dir)
    dir = dir / np.max(dir)
    return dir
# Read the images in grayscale
blobs = cv2.imread('GradWorld-blobs.jpg', cv2.IMREAD_GRAYSCALE)
oval = cv2.imread('GradWorld-oval.jpg', cv2.IMREAD_GRAYSCALE)
triangle = cv2.imread('GradWorld-triangle.jpg', cv2.IMREAD_GRAYSCALE)

blobs_fix = image_pre_process(blobs)
oval_fix = image_pre_process(oval)
triangle_fix = image_pre_process(triangle)

fig, axes = plt.subplots(2,3, figsize=(10, 10))
axes = axes.flatten()
for img, ax, title in zip([blobs,oval,triangle,blobs_fix,oval_fix,triangle_fix],
                          axes, 
                          ["Blobs","Oval","Triangle","","Preprocess",""]):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
# Practial session code

# The support function:
def support(comp_kernel,confidence):
    '''
    This function is general now and returns the support for each pixel getting 
    each of the possible labels. E.g., supp[0] is a matrix same shape as our image,
    and the value in each of its cells represents the support for assigning label
    0 to the corresponding pixel in the image.
    '''
    nLabels, nRows, nColumns = confidence.shape
    supp = np.zeros_like(confidence)
    for label in range(nLabels):
        supp[label] = cv2.filter2D(confidence[label], ddepth=cv2.CV_64F, kernel=comp_kernel).astype(np.float)
    return supp

# The confidence update rule:
def update(curr_confidence, comp_kernel):
    nLabels, nRows, nColumns = curr_confidence.shape
    res = np.zeros_like(curr_confidence)
    res_denominator = np.zeros((nRows,nColumns))
    supp = support(comp_kernel,curr_confidence)
    for label in range(nLabels):
        res[label] = curr_confidence[label]*supp[label]
        res_denominator += res[label]
    res = res / res_denominator
    return res

# The average local consistency:
def average_local_consistency(confidence, comp_kernel):
    nLabels, nRows, nColumns = confidence.shape
    res_denominator = np.zeros((nRows,nColumns))
    supp = support(comp_kernel,confidence)
    for label in range(nLabels):
        res_denominator += confidence[label]*supp[label]
    alc = np.sum(res_denominator) #the average local consistency
    return alc
    
# Set the final assignment for each pixel as the label with the highest probability
def choose_label(confidence):
    nLabels, nRows, nColumns = confidence.shape
    segmentation_array = np.zeros((nRows, nColumns))
    for i in range(nRows):
        for j in range(nColumns):
            segmentation_array[i,j] = np.argmax(confidence[:,i,j])
    return segmentation_array
# The relaxation labeling algorithm:
# get list and number , return the index of the closest value to number in list 
def closest(list, Number):
    aux = []
    for valor in list:
        aux.append(abs(Number-valor))
    return aux.index(min(aux))

# get array and int-list , change array's values to closest values in list 
def close_matrix (image,val_segments):
    for idx , val in np.ndenumerate(image):
        image[idx] = val_segments[closest(val_segments,val)]
    return image 

#input image, window size ,range of image (after preprocessing the image), how many pixels to include in computation 
#output: list of numbers, each value is median of a top segment 
def group_pixels(image,window_size,range_of_labels,include):
    image_size = image.shape[0] * image.shape[1]
    unique, counts = np.unique(image, return_counts=True)
    arr = np.zeros(range_of_labels + 1,int)
    # arr is unique&counts data without jumps 
    
    for val , count in zip(unique,counts):
        arr[int(val)] = count # i like to live dangerously 
    sum_of_window = np.zeros_like(arr)
    for i,_ in enumerate(arr):
        sum_of_window[i] = sum(arr[ i - window_size : i + window_size])
    
    # make new group and remove its pixels from arr (and update sum_of_window) 
    ret = []
    count_pixels = 0
    while count_pixels < include * image_size : 
        max = np.argmax(sum_of_window)
        ret.append(max)
        count_pixels += sum_of_window[max]
        arr[ max - window_size : max + window_size ] = 0 
        for i , _ in enumerate(arr):
            sum_of_window[i] = sum(arr[ i - window_size : i + window_size])
    return ret 

#input: image, list of top segments values 
#output: set high probability to closest label 
def set_initial_confidence(image,val_segments):
    image = close_matrix(image,val_segments)
    nLabels = len(val_segments)
    initial_confidence = np.zeros((nLabels,)+image.shape).astype(np.float)
    
    for label in range(nLabels):
        initial_confidence[label][image == val_segments[label]] = 1 - 0.0001 * nLabels
        initial_confidence[label][image != val_segments[label]] = 0.0001 * nLabels
    return initial_confidence

def RelaxationLabeling(image,window_size,include,comp_kernel,epsilon,range_of_labels):
    image = (image_pre_process(image) * range_of_labels).astype(int)
    values_of_segments = group_pixels(image,window_size,range_of_labels,include)
    curr_conf = set_initial_confidence(image,values_of_segments) # This is P for iteration k=0
    k = 0 # counts the iteration number
    while True:
        next_conf = update(curr_conf, comp_kernel)
        diff = abs(average_local_consistency(curr_conf, comp_kernel) - 
                   average_local_consistency(next_conf, comp_kernel))

        curr_conf = next_conf
        k = k + 1
        if diff < epsilon:
            break
    return choose_label(next_conf)
# Use your algorithm to segment the 3 grayscale images
comp_kernel = np.array([[1/9,1/9,1/9],
                        [1/9,1/9,1/9],
                        [1/9,1/9,1/9]])
window_size = 15 #15
include = 0.8 #8
epsilon = 0.01
range_of_labels = 360
segmented_images = [RelaxationLabeling(img,window_size,include,comp_kernel,epsilon,range_of_labels) for img in [blobs,oval,triangle]]

# Plots your results (3 images)
fig, axes = plt.subplots(1,3, figsize=(10, 10))
for img, ax, title in zip(segmented_images,
                          axes, 
                          ["Blobs Segmented","Oval Segmented","Triangle Segmented"]):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)