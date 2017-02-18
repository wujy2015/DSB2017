import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pylab as pl
from skimage import measure, morphology, segmentation
import scipy.ndimage as ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#import lmdb
import deepdish as dd

import os
old_path = os.getcwd()
caffe_root = "/home/jwu/caffe/caffe"
os.chdir(caffe_root + "/python")
import caffe

# Some constants
INPUT_FOLDER = '/home/jwu/Downloads/stage1_full/stage1/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()


def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)
def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    print "real_resize_factor"
    print real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def generate_markers(image):
    # Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    # Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    # Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((253, 253), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed


def seperate_lungs(image):
    # Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)

    # Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    # Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    # Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3, 3))
    outline = outline.astype(bool)

    # Performing Black-Tophat Morphology for reinclusion
    # Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    # Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    # Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    # Close holes in the lungfilter
    # fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5, 5)), iterations=3)

    # Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000 * np.ones((253, 253)))

    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed

def seperate_lungs_3d(image3D):
    segmented_3d = []; lungfilter_3d = []; outline_3d = []; watershed_3d = []; sobel_gradient_3d = []
    marker_internal_3d = []; marker_external_3d = []; marker_watershed_3d = []
    for i in range(len(image3D)):
        segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed = seperate_lungs(image3D[i])
        segmented_3d.append(segmented)
        lungfilter_3d.append(lungfilter)
        outline_3d.append(outline)
        watershed_3d.append(watershed)
        sobel_gradient_3d.append(sobel_gradient)
        marker_internal_3d.append(marker_internal)
        marker_external_3d.append(marker_external)
        marker_watershed_3d.append(marker_watershed)
    return np.array(marker_internal_3d, dtype=np.int16), np.array(marker_external_3d, dtype=np.int16), np.array(marker_watershed_3d, dtype=np.int16)\
            ,np.array(sobel_gradient_3d, dtype=np.int16), np.array(watershed_3d, dtype=np.int16), np.array(outline_3d, dtype=np.int16) \
            , np.array(lungfilter_3d, dtype=np.int16), np.array(segmented_3d, dtype=np.int16)



df = pd.read_csv('/home/jwu/Downloads/dsb2017/data/stage1_sample_submission.csv')
patients_test = df['id'].tolist()
patients_train = [p for p in patients if p not in patients_test]
#X = np.zeros((len(patients_train), 240, 253, 253), dtype=np.uint16)
#y = np.zeros(len(patients_train), dtype=np.int64)
#map_size = X.nbytes * 10
#del X

#env = lmdb.open('mylmdb2', map_size=int(1e14))

df = pd.read_csv('/home/jwu/Downloads/dsb2017/data/stage1_labels.csv')
y_test = [ df['cancer'][df.id == x] for x in patients_test]
y_train = [ df['cancer'][df.id == x] for x in patients if x not in patients_test]

#with env.begin(write=True) as txn:
inputDataTrain = []
yTrain = []
for i in range(len(patients_train)):
    p = patients_train[i]
#for i in range(len(patients_test)):
#    p = patients_test[i]
    patient = load_scan(INPUT_FOLDER + p)
    patient_pixels = get_pixels_hu(patient)
    pix_resampled, spacing = resample(patient_pixels, patient, [1, 1, 1])
    lx, ly, lz = pix_resampled.shape
    if lx < 240 and ly < 253 and lz < 253:
        continue
    pix_resampled_crop = pix_resampled[(lx-240)/2:(240-lx)/2,(ly-253)/2:(253-ly)/2,(lz-253)/2:(253-lz)/2]
    nx, ny, nz = pix_resampled_crop.shape
    if nx != 240 or ny != 253 or nz != 253:
        print "croping problem: ", pix_resampled.shape, " ", pix_resampled_crop.shape
        continue
    marker_internal_3d, marker_external_3d, marker_watershed_3d, sobel_gradient_3d, watershed_3d, outline_3d, lungfilter_3d, segmented_3d = seperate_lungs_3d(pix_resampled_crop)

    try:
        label = int(df['cancer'][df.id == p])
        print "label: ", label
    #   datum.label = int(df['cancer'][df.id == p])
    except:
        print "label not found: ", p
        continue

    #str_id = '{:08}'.format(i)
    # The encode is only essential in Python 3
    #txn.put(str_id.encode('ascii'), datum.SerializeToString())
    #inputDataTrain.append(pix_resampled_crop)
    #dd.io.save('/home/jwu/caffe/caffe/python/training_data3/' + p + '.h5', {'data': np.array([pix_resampled_crop]), 'label': np.array([label])}, compression=None)
    dd.io.save('/data_center/marker_internal_3d/' + p + '.h5',
           {'data': np.array([pix_resampled_crop]), 'label': np.array([label])}, compression=None)
    dd.io.save('/data_center/marker_external_3d/' + p + '.h5',
           {'data': np.array([pix_resampled_crop]), 'label': np.array([label])}, compression=None)
    dd.io.save('/data_center/marker_watershed_3d/' + p + '.h5',
           {'data': np.array([pix_resampled_crop]), 'label': np.array([label])}, compression=None)
    dd.io.save('/data_center/sobel_gradient_3d/' + p + '.h5',
           {'data': np.array([pix_resampled_crop]), 'label': np.array([label])}, compression=None)
    dd.io.save('/data_center/watershed_3d/' + p + '.h5',
           {'data': np.array([pix_resampled_crop]), 'label': np.array([label])}, compression=None)
    dd.io.save('/data_center/outline_3d/' + p + '.h5',
           {'data': np.array([pix_resampled_crop]), 'label': np.array([label])}, compression=None)
    dd.io.save('/data_center/lungfilter_3d/' + p + '.h5',
           {'data': np.array([pix_resampled_crop]), 'label': np.array([label])}, compression=None)
    dd.io.save('/data_center/segmented_3d/' + p + '.h5',
           {'data': np.array([pix_resampled_crop]), 'label': np.array([label])}, compression=None)
#    dd.io.save('/home/jwu/caffe/caffe/python/testing_data/' + p + '.h5', {'data': pix_resampled_crop}, compression=None)
"""
inputDataTrain = np.array(inputDataTrain, dtype=np.int16)
yTrain = np.array(yTrain, dtype=np.int16)
dd.io.save('test.h5', {'data': inputDataTrain, 'label': yTrain}, compression=None)
"""