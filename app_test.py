from flask import Flask,render_template,request,url_for,send_from_directory
import eval
import os
import shutil
import sys
from PIL import Image
import pytesseract
from ocr_for import ocr_image_and_split_multiwords

app = Flask(__name__)


#Get the directory off the 'app_test.py'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/',methods=["POST","GET"])
def file_uploader():
	return render_template("save_upload.html")


#To upload the images ,extract the text and save in the 'images' folder
@app.route('/upload',methods=["POST"])
def index():

	#Make a folder named 'images' where uploaded images will be stored
	target = os.path.join(APP_ROOT,'images/')
	
	if not os.path.isdir(target):
		os.mkdir(target)
	else:
		shutil.rmtree(target)
		os.mkdir(target)


	images_captions =[]#Stores the caption -i.e the final text extracted from images uploaded
	try:

		image_names =[]#Stores the image names that will be uploaded
		#caption_names=[]#Stores the corresponding text we want as caption below the image when it's clicked
		
		upload_image_list = request.files.getlist('file')#Get the names of the images uploaded through forms and request

		for image in upload_image_list:
			
			
			
			filename = image.filename
			image_names.append(filename)#This list will be passed through render template



			destination = '/'.join([target,filename])#File location for storing the image in the 'images' folder
			image.save(destination)#Save the image in the images folder

			

		#===============================================================
		eval.main()#Generates the border_image and txt of 8 co-ordinate
		#===============================================================

		#Generating list that stores the all word_box_co-ordinates by reading the 4-co-ordinate txt 
		
		img_with_noise_path  = os.path.join(APP_ROOT,'images/')

		plst_word_box_coordinates_path = os.path.join(APP_ROOT,'output_label/')

		uploaded_images = os.listdir(img_with_noise_path)

		pstr_cropped_images_folder_path = os.path.join(APP_ROOT,'output_crop')

		pstr_intermediate_output_folder_path = os.path.join(APP_ROOT,'intermediate')

		inp = 'input.txt'#Hard_coded

		out = 'output'#Hard_coded


		#Calling the ocr on each image and generating the image and caption and storing it in 'images_captions' with each entry as tuple
		for image in uploaded_images:
			
		    word_box_coordinates= []
		    

		    pth = plst_word_box_coordinates_path + image.split('.')[0] +'.txt'#The code will fail if the image file name will contain another '.'
		    
		    with open(pth) as f:	
		        for line in f:
		            temp = line.split(' ')
		            box = []
		            for i in temp:
		                box.append(int(i))
		            word_box_coordinates.append(box)

		    img_path = img_with_noise_path + image

		    image_temp,caption_temp =image,ocr_image_and_split_multiwords(word_box_coordinates,img_path,pstr_cropped_images_folder_path,pstr_intermediate_output_folder_path,inp,out)
		    
		    images_captions.append((image_temp,caption_temp))


		    

		


			
		return render_template("home_2.html",image_caption=images_captions)

	except:
		#When no image is selected and send button is clicked 
		return render_template('save_upload.html')



#Called  in 'home_2.html' in 'img' tag'src=url_for()' to send the image file location
@app.route('/upload/<filename>')
def send_image(filename):

	


	return send_from_directory("output_images",filename)



if __name__ == '__main__':
	app.run(debug = True)


