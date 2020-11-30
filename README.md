# Image Orientation Correction
Script for correcting rotated images of an arbitrary text. It is based on the algorithm that was written by user flamelite here https://stackoverflow.com/questions/56854176/is-there-a-way-i-can-detect-the-image-orientation-and-rotate-the-image-to-the-ri

## Requirements
This code depends on NumPy and OpenCV
To install these requirements you can use:

 `pip install -r requirements.txt`

In order to run this code, use: 

`python main.py path_to_your_image`

The resulting image will be in the same folder as the code, or you can specify an optional argument output_path to save it in a specific folder.

`python main.py path_to_your_image output_path` 
