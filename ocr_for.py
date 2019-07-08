#The script works on single image and is imported in the 'app_test.py' , returns the text string of combined boxes detected in the images displayed on the webpage
#Order of the boxes and their text is not  maintained

#=========================================INPUT================================================
#The input to the below function takes 
#1->plst_word_box_coordinates - [[x01,y01,x02,y02],[x11,y11,x12,y12],....[...]] (stored in the 'output_label' folder in the '.txt' file) (4 coordinates of a box)
#2->img_with_noise - takes path of the original image without the borders around text
#3->pstr_cropped_images_folder_path - path of the cropped images of the bounding boxes where text is detected and stores each cropping in this folder path
#4->pstr_intermediate_output_folder_path - path of the intermediate folder where 2 files are created 'input.txt' and 'output.txt'
#5->pstr_cropped_images_list_file_name - this is the 'input.txt' and stores the path of all the cropped images created in between
#6->pstr_tesseract_results_file_path - path of the 'output.txt' file where the text extracted from the cropeed images are stored with a page seperator
#==============================================================================================
import cv2
import subprocess
import numpy as np
from subprocess import Popen, PIPE
import pandas as pd
import os
import shutil
import matplotlib.path as mplPath
import re


def ocr_image_and_split_multiwords(plst_word_box_coordinates,img_with_noise,pstr_cropped_images_folder_path,pstr_intermediate_output_folder_path,pstr_cropped_images_list_file_name,pstr_tesseract_results_file_path):
    llst_objects = []
    lint_counter = 0
    lint_obj_counter = 0
    lint_c = 0
    llst_ocr_text_info = []

    # color for border+
    llst_color = [255, 255, 255]

    parr_noise_removed_image = cv2.imread(img_with_noise)
    lstr_cropped_images_list_file_name = pstr_intermediate_output_folder_path+"/"+pstr_cropped_images_list_file_name
    lstr_tesseract_results_file_path = pstr_intermediate_output_folder_path+"/"+pstr_tesseract_results_file_path

    #create folder for storing cropped patches
    lstr_cropped_img_folder_path = pstr_cropped_images_folder_path

     
    #if there is already a folder containing cropped pathches delete this directory and it's files and make a new one with the same name
    if not os.path.exists(lstr_cropped_img_folder_path):
        os.makedirs(lstr_cropped_img_folder_path)
    else:
        shutil.rmtree(lstr_cropped_img_folder_path)
        os.mkdir(lstr_cropped_img_folder_path)



    #Stores the paths of cropped images in the 'input.txt' and ocr is performed
    with open(lstr_cropped_images_list_file_name, 'w') as out_file: 
        for llst_word_box_coordinate in plst_word_box_coordinates:
            larr_cropped_image = parr_noise_removed_image[llst_word_box_coordinate[1]:llst_word_box_coordinate[3], llst_word_box_coordinate[0]:llst_word_box_coordinate[2]]
            
            # added border to cropped patch
            larr_img_with_border = cv2.copyMakeBorder(larr_cropped_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=llst_color)
            lstr_cropped_image_name = lstr_cropped_img_folder_path + '/%d.jpg' % lint_counter
            cv2.imwrite(lstr_cropped_image_name, larr_img_with_border)
            
            out_file.write("%s\n" % lstr_cropped_image_name)
            lint_counter = lint_counter + 1
    
    lstr_command = "tesseract '"+lstr_cropped_images_list_file_name+"' '"+lstr_tesseract_results_file_path+"' "+" --psm 7 -c page_separator='[PAGE SEPARATOR]'"        

    try:
        lobj_process = Popen(lstr_command, stderr=PIPE, stdout=PIPE, shell=True)           
    except Exception:
        raise

    #===================================================================
    # check return code of process
    # if return code = 0 then process executed with no errors
    # otherwise process is failed
    #===================================================================
    lbyt_output, lbyt_error = lobj_process.communicate()
    if lobj_process.returncode != 0: 
        lstr_info = "tesseract failed with error code: %d and error: %s" % (lobj_process.returncode, lbyt_error.decode("utf-8"))
        print(lstr_info) 

    else:
        #===========================================================================
        # read output file created by tesseract and return predicted text
        #===========================================================================
        llst_ocr_txt = []
        llst_cropped_images_paths = []
        with open(lstr_tesseract_results_file_path+".txt", "r") as read_file:
            llst_ocr_txt = read_file.read()
            
        with open(lstr_cropped_images_list_file_name, "r") as read_file:
            llst_cropped_images_paths = read_file.read()
        
        llst_cropped_images_paths = llst_cropped_images_paths.split("\n")
        llst_ocr_text_info = llst_ocr_txt.split("[PAGE SEPARATOR]")
        llst_ocr_text_info = [x.strip() for x in llst_ocr_text_info[:-1]]
        #llst_ocr_text_info-contains the list of all text extracted from each box 

        caption_str = ' '.join(llst_ocr_text_info)#convert the texts into a combine string without order
        
        return caption_str 
