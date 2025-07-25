This project uses deep learning to classify houseplant species and detect plant diseases from images using two fine-tuned VGG16 models.

📁 Directory Structure

├── input_planttype/                       # Place images for plant type classification here

├── input_plantdisease/                    # Place images for plant disease detection here

├── main.py                                # Main script to run predictions

├── modified_vgg16.py                      # Custom VGG16 model architecture

├── otsu_canny.py                          # Applies Otsu thresholding + Canny edge detection

├── image_augmentation.py                  # Performs image rotation and flipping for augmentation

├── planttype.keras                        # Pretrained model for plant type classification

├── plantdisease.keras                     # Pretrained model for plant disease detection

├── Datset(disease)                        # Dataset includes total of 7152 plant disease images

├── Datset(disease)_otsucanny              # Dataset includes total of 7152 plant disease images processed using otsu thresholding followed by canny edge detection

├── Dataset (type)                         # Dataset includes total of 1500 plant type images

├── Dataset (type)_otsucanny_augmented     # Dataset includes a total of 6000 plant type images that were processed using Otsu thresholding and canny edge detection, followed by augmentation techniques 

├── Dataset (type)_augmetned               # Dataset includes total of 6000 plant type images after augmentation


🚀 How to Use
1. Classify Plant Type
Place all relevant images inside the input_planttype folder.

2. Classify Plant Disease
Place all relevant images inside the input_plantdisease folder.

3. Run the Classifier
Simply execute the following command:
python main.py

5. Results
The program will generate predictions for each image.
Outputs include: Predicted Class, Confidence Score, Plant Care Tips (for disease cases)

🧠 Model Architecture

The model used is a modified VGG16 CNN architecture.

The code defining the network can be found in modified vgg16.py.

🛠 Utility Scripts
1. otsucanny.py
Applies Otsu thresholding to images, then uses the resulting threshold to apply Canny edge detection.
Use this to analyze edge features of plant images.

2. image_augmentation.py
Performs image augmentation including:
Random rotations
Horizontal and vertical flips
Useful for expanding your dataset before training or evaluation.

📌 Notes

Image input formats should be .jpg, .jpeg, or .png.
Ensure consistent naming for files and avoid using special characters in filenames.
