import os

import shutil
import os
import time

def copy_images(image_paths, destination_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    for image_path in image_paths:
        try:
            # Copy the image to the destination folder
            shutil.copy(image_path, destination_folder)
            print(f"Copied: {image_path}")
        except FileNotFoundError:
            print(f"File not found: {image_path}")
        except Exception as e:
            print(f"Error copying {image_path}: {e}")

# Example usage
image_paths = [
]

f = open("path.txt", "r")
for each in f:
    each = each.replace("\n", "")
    image_paths.append(each)

destination = r"\Users\lovro\Desktop\OCR-LLM-Enhancer\testing"

copy_images(image_paths, destination)

mypath = "/Users/lovro/Desktop/OCR-LLM-Enhancer/dataset"

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for each in onlyfiles:
    each = each.replace("\n", "")
    print("/Users/lovro/Desktop/OCR-LLM-Enhancer/images/" + each)



    '''
    /Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353582.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354789.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354158.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354159.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354788.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353583.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353581.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353378.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353584.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353751.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353817.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353585.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353578.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354149.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353395.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353425.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354160.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354148.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353708.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353720.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05355225.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354477.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353495.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354476.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354112.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353709.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354474.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353496.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353497.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354475.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05355227.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354111.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353722.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353646.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353450.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354470.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353451.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353647.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353447.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354467.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353452.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353724.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354046.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354052.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353701.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354898.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353648.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354053.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354045.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353702.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353448.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354469.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354468.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353449.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353703.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354044.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353539.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354054.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354097.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354083.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354478.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354479.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354082.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354096.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354055.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353712.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353710.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354043.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354057.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354902.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353466.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353467.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354081.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354056.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353574.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354780.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354186.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353367.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353401.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353575.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353549.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353403.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05355067.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354782.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354209.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354779.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354792.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354157.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354793.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354778.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354220.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05356409.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354791.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05353404.jpg
/Users/lovro/Desktop/OCR-LLM-Enhancer/images/Z05354790.jpg
'''
    