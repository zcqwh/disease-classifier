#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:52:23 2022

@author: nana
"""
import os, time, h5py
import numpy as np
from . import image_processing
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def bin_2_rtdc(bin_path):
    t1 = time.time()
    ###############Parameters##########################
    filter_len = True
    len_min = 10
    len_max = 300
    
    filter_area = False
    area_min = 10
    area_max = 50
    
    filter_n = True
    nr_contours = 100
    
    pixel_size = 0.8
    noise_level = 8
    bg_intensity = 16381
    ###############Parameters##########################
    
    path_target = os.path.dirname(bin_path) + os.sep + os.path.basename(bin_path).split(".bin")[0] + ".rtdc" #filename of resulting file
    #check if the file that should be written may already exists (e.g. after runnig script twice)
    if os.path.isfile(path_target):
        print("Following file already exists and will be overwritten: "+path_target)
        #delete the file
        os.remove(path_target)
    
    
    #load binary
    binary = np.fromfile(bin_path, dtype='>H')
    n,w,h = binary[1],binary[3],binary[5]
    images = binary[6:].reshape(n,h,w)
    
    images = images.astype(np.float32) #for conversion to unit8, first make it float (float32 is sufficient)
    factor = 128/bg_intensity
    images = np.multiply(images, factor)
    images = images.astype(np.uint8)

    #Cell can be darker and brighter than background->Subtract background and take absolute
    images_abs = images.astype(np.int32)-128 #after that, the background is approx 0
    images_abs = abs(images_abs).astype(np.uint8) #absolute. Neccessary for contour finding
    

    #Initialize lists for all properties
    index_orig,images_save,masks,contours,\
    pos_x,pos_y,size_x,size_y,\
    bright_avg,bright_sd,\
    area_um,area_orig,area_hull,\
    area_ratio,circularity,inert_ratio_raw = \
    [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    
    t1 = time.time()
    #get all images located in one particular folder
    for img_index in range(len(images)):
        #load image
        image = images[img_index]
        ####
        #image, factor = uint16_2_unit8(image) 
        image_abs = images_abs[img_index]
                
        #get list of conoturs and masks in image using the superposition
        contours_,masks_ = image_processing.get_masks_ipac(image_abs, noise_level,filter_len, len_min, len_max, 
                                             filter_area, area_min, area_max, filter_n, nr_contours)
        del image_abs
        
        for contour,mask in zip(contours_,masks_):
            output = image_processing.get_boundingbox_features(image,contour,pixel_size)
    
            if type(contour)!=np.ndarray or np.isnan(output[0]) or np.isnan(output[1]):
                index_orig.append(img_index)
                images_save.append(np.zeros(shape=image.shape,dtype=np.uint8))
                masks.append(np.zeros(shape=image.shape))
                contours.append(np.nan)
                pos_x.append(np.nan)
                pos_y.append(np.nan)
                size_x.append(np.nan)
                size_y.append(np.nan)
    
                bright_avg.append(np.nan)
                bright_sd.append(np.nan)
                area_orig.append(np.nan)
                area_hull.append(np.nan)
                area_um.append(np.nan)
                area_ratio.append(np.nan)
                circularity.append(np.nan)
                inert_ratio_raw.append(np.nan)
    
            else:
                index_orig.append(img_index)
                images_save.append(image)
                masks.append(mask)
                contours.append(contour)
                pos_x.append(output[0])
                pos_y.append(output[1])
                size_x.append(output[2])
                size_y.append(output[3])
                
                output = image_processing.get_brightness(image,mask)
                bright_avg.append(output["bright_avg"])
                bright_sd.append(output["bright_sd"])
                
                output = image_processing.get_contourfeatures(contour,pixel_size)
                area_orig.append(output["area_orig"])
                area_hull.append(output["area_hull"])
                area_um.append(output["area_um"])
                area_ratio.append(output["area_ratio"])
                circularity.append(output["circularity"])
                inert_ratio_raw.append(output["inert_ratio_raw"])
    
    t2 = time.time()
    dt = t2-t1
    print("Required time to compute features: " +str(np.round(dt,2) )+"s ("+str(np.round(dt/len(images_save)*1000,2) )+"ms per cell)")
    
    
    #remove events where no contours were found (pos_x and pos_y is nan)
    ind_nan = np.isnan(pos_x)
    ind_nan = np.where(ind_nan==False)[0]
    if len(ind_nan)>0:
        index_orig = list(np.array(index_orig)[ind_nan])
        pos_x = list(np.array(pos_x)[ind_nan])
        pos_y = list(np.array(pos_y)[ind_nan])
        size_x = list(np.array(size_x)[ind_nan])
        size_y = list(np.array(size_y)[ind_nan])
    
# =============================================================================
#         images_save = list(np.array(images_save,dtype=np.uint8)[ind_nan])
#         masks = list(np.array(masks,dtype=np.bool_)[ind_nan])   
# =============================================================================
        images_save = [images_save[index] for index in ind_nan]
        masks = [masks[index] for index in ind_nan]
        
        contours = list(np.array(contours)[ind_nan])
        bright_avg = list(np.array(bright_avg)[ind_nan])
        bright_sd = list(np.array(bright_sd)[ind_nan])
        area_orig = list(np.array(area_orig)[ind_nan])
        area_hull = list(np.array(area_hull)[ind_nan])
        area_um = list(np.array(area_um)[ind_nan])
        area_ratio = list(np.array(area_ratio)[ind_nan])
        circularity = list(np.array(circularity)[ind_nan])
        inert_ratio_raw = list(np.array(inert_ratio_raw)[ind_nan])
        
        #Save images and corresponding pos_x and pos_y to an hdf5 file for AIDeveloper
        images_save = np.array(images_save)
        masks = np.array(masks)
        
        maxshape_img = (None, images_save.shape[1], images_save.shape[2])
        maxshape_mask = (None, masks.shape[1], masks.shape[2])
        
        #Create rtdc_dataset; valid feature names can be found via dclab.dfn.feature_names
        hdf = h5py.File(path_target,'a')
        dset = hdf.create_dataset("events/image", data=images_save, dtype=np.uint8,maxshape=maxshape_img,fletcher32=True,chunks=True)
        dset.attrs.create('CLASS', np.string_('IMAGE'))
        dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
        dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
    
        dset = hdf.create_dataset("events/mask", data=masks, dtype=np.uint8,maxshape=maxshape_mask,fletcher32=True,chunks=True)
        dset.attrs.create('CLASS', np.string_('IMAGE'))
        dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
        dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
        
        hdf.create_dataset("events/index_online", data=index_orig,dtype=np.int32)
        hdf.create_dataset("events/pos_x", data=pos_x, dtype=np.int32)
        hdf.create_dataset("events/pos_y", data=pos_y, dtype=np.int32)
        hdf.create_dataset("events/size_x", data=size_x, dtype=np.int32)
        hdf.create_dataset("events/size_y", data=size_y, dtype=np.int32)
        
        hdf.create_dataset("events/bright_avg", data=bright_avg, dtype=np.float32)
        hdf.create_dataset("events/bright_sd", data=bright_sd, dtype=np.float32)
    
        hdf.create_dataset("events/circ", data=circularity, dtype=np.float32)
        hdf.create_dataset("events/inert_ratio_raw", data=inert_ratio_raw, dtype=np.float32)
        hdf.create_dataset("events/area_ratio", data=area_ratio, dtype=np.float32)
        hdf.create_dataset("events/area_msd", data=area_orig, dtype=np.float32)
        hdf.create_dataset("events/area_cvx", data=area_hull, dtype=np.float32)       
        hdf.create_dataset("events/area_um", data=area_um, dtype=np.float32)       
    
        #Adjust metadata:
# =============================================================================
#                 #"experiment:event count" = Nr. of images
#                 hdf.attrs["experiment:run index"] = m_number
#                 m_number += 1 #increase measurement number 
# =============================================================================
        hdf.attrs["experiment:event count"] = images_save.shape[0]
        #hdf.attrs["experiment:sample"] = condition #Blood draw date
        hdf.attrs["imaging:pixel size"] = pixel_size
        hdf.attrs["experiment:date"] = time.strftime("%Y-%m-%d")
        hdf.attrs["experiment:time"] = time.strftime("%H:%M:%S")

        hdf.attrs["imaging:roi size x"] = images_save.shape[2]
        hdf.attrs["imaging:roi size y"] = images_save.shape[1]
        hdf.attrs["online_contour:bin kernel"] = 2*int(3)+1
        hdf.attrs["online_contour:bin threshold"] = noise_level

        hdf.close()
    
    t2 = time.time()
    print("bin. to rtdc.:",np.round((t2-t1),4))
    
    return path_target



