# PSENet_Image_to_text_flask_implementaion
Upload the images in 'jpeg' ,'jpg' ,'png' format containing text and obtain all the text in the image present.

## Project Description
To detect text ,Progressive scale expansion(PSENet) model has beed used, which has been trained over 1000 images of forms , works best on images with white background and black text font. To detect images of various aspects ,one needs to trained the PSENet model and copy the obtained checkpoint in the checkpoints folder above.

A flask API has been provided to evaluate the results obtained from the trained checkpoints

## Required Libraries

- tensorflow
- tesseract
- Flask
- Opencv
- numpy
- pandas
- matplotlib
- subprocess
- Pillow

##  Scripts Imported
1. app_test.py 
   - Main flask file
 
2. eval.py 
   - Genarates the bordered images and 8 co-ordinate txt in 'output_images' and converts it into 4 co-ordinate txt and stores it in 'output_label'

3. ocr_for.py 
   - Generates the caption from the detected boxes ,generates cropping and store it in 'output_crop' folder and ocr text in 'intermediate' folder in the 'output.txt'

## Folder contents
1. checkpoint 
   - Contain the trained checkpoints from deep_learning models(Here PSEnet is used that is trained over 1000 images)

2. images 
   - Will store the uploaded images of the user

3. intermediate 
   - will contain the 'input.txt' of cropped patches and 'output.txt'

4. nets 
   - Needed to run eval.main() 

5. output_crop 
   - Contain the crop pathches of the last image processed from the 'images' folder

6. output_images 
   - Contain the images(that will be displayed) with their 8 co-ordinate txt (both have same name)

7. output_label 
   - Contains only the 4 co-ordinate text evaluated from 8 co-ordinate text

8. pse 
   - Needed to run eval.main()

9. templates 
   - contain the html rendered files 

10. utils 
    - Needed to run eval.main()

## How to use 

Download the checkpoint from this [link](https://drive.google.com/open?id=1jJATwSwc9L6lbRAJRR7Mr0mGeTXJE_kH),
then copy this checkpoint in the checkpoint folder after cloning the whole repository.
   - Local server-
     - Clone the repository , and run the 'app_test.py' file
   - Host the files
     - Clone the repository and commit or upload all the files to the server,with your respective domain name acting as the website. 

## Further improvements
The better the checkpoint, better will be the results on the uploaded images

If we want to detect objects or recognise face , we need to only change the checkpoints in the checkpoint folder and the evaluation procedure of the respective model used.

To accomodate various forms of images , we can process every image with OpenCV such that all the text is converted to black and remaining background to white and then the model can be applied.
