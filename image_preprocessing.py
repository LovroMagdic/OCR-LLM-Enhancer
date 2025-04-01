import cv2 
import pytesseract
import os
import numpy as np
import math
from pathlib import Path
from cv2 import dnn_superres

# creating folder to store results for each step of preprocessing
def createFolders():
    os.mkdir("dataset_adaptive")
    os.mkdir("dataset_blur")
    os.mkdir("dataset_contour")
    os.mkdir("dataset_copy")
    os.mkdir("dataset_deskewed")
    os.mkdir("dataset_final")
    os.mkdir("dataset_processed_deskew")
    os.mkdir("dataset_thick")
    os.mkdir("dataset_thin")
    os.mkdir("dataset_denoise")
    os.mkdir("ocr")
    os.mkdir("dataset_upscaled")

#function used for rotating image considering image center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

#function for grayscaling image
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#function used for thinning font by using multiple cv2 builtin functions
def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,1),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

#function used for creating thicker font by using multiple cv2 builtin functions
def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

#here starts main script
dir = os.getcwd()
folders = []
for each in os.scandir(dir):
    folders.append(str(each))
if "<DirEntry 'dataset_final'>" not in folders:
    createFolders()
dir = os.path.join(dir,"dataset")
dir = dir.replace("\\", "/")
arr = [] # sadrzi imena svih dataset slika

#we are saving each file stored in dataset in arr array so that is easier to iterate after
for filename in os.scandir(dir):
    if filename.is_file():
        arr.append(filename.path)

#every each in for loop represents an iterated image in folder
#saving images to use them for houghline transform, line detection
for each in arr:
    tmp_img = cv2.imread(each)

    gray_image = grayscale(tmp_img)
    thresh, im_bw = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    each = each.replace("dataset", "dataset_processed_deskew")
    cv2.imwrite(each, im_bw)

#saving images on which rotation will actually take place
for each in arr:

    image = cv2.imread(each, 0)
    each = each.replace("dataset","dataset_copy")
    cv2.imwrite(each, image)

#line detection and rotation script
#real_img is an image which is actually being rotated and image is dummy image which we use for line detection and deskew angle
for each in arr:
    arr1 = []
    each = each.replace("dataset", "dataset_copy")
    real_img = cv2.imread(each)
    each=each.replace("dataset_copy", "dataset_processed_deskew")
    img = cv2.imread(each)
    arr2 = []
    global_angle = 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    #caluclating angles found on image, since we found alot of angles and some of them arent useful we discard some afterwards
    for r_theta in lines:
        arr1 = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr1

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        if ((x2-x1) != 0):
            res1 = (y2-y1)/(x2-x1)
            res2 = y1-res1*x1
        else:
            res1 = 0
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        k2 = res1
        angle = math.tan(abs((k2-0)/(1+k2*0)))
        res = math.degrees(math.atan(angle))
        
        #we consider only angles between 0-10, by doing that we get better results on whole dataset
        if res > 0 and res < 10:
            arr2.append(res)
            global_angle += res

    each = each.replace("dataset_processed_deskew","dataset_draw")
    cv2.imwrite(each, img)
    each = each.replace("dataset_draw","dataset_processed_deskew")
    
    #global_angle is final calculated angle for which particular image in question is deskewed
    global_angle = global_angle/len(arr2)
    
    #calling rotating function with param (actual image rotated, angle of rotation)
    deskew_img = rotateImage(real_img, -1 * global_angle)

    each=each.replace("dataset_processed_deskew", "dataset_deskewed")
    cv2.imwrite(each, deskew_img)

# calculating threshold for which we get best results
ar = []
array = []

#iterating through threshold value of 100 - 235 and calculating minimal remainder like that we get most optimal threshold which gives us best contour
for each in arr:
    each = each.replace("dataset","dataset_deskewed")
    for i in range(100, 235, 5): # i == best_thresh
        image = cv2.imread(each)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, i, 255, cv2.THRESH_BINARY)
        # cv2.imwrite('demo/test-bw.jpg', thresh)

        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        image_copy = image.copy()

        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # cv2.imwrite("image-draw.jpg", image_copy)

        c = max(contours, key = cv2.contourArea)
        # print(cv2.contourArea(c))
        x,y,w,h = cv2.boundingRect(c)
        # print(w*h)

        ar.append([cv2.contourArea(c), w*h, i])
        array.append(int(w*h)-int(cv2.contourArea(c)))

    index = array.index(min(array))
    # print(ar[index])

    i = int(ar[index][2])
    # each = each.replace("dataset_deskewed","dataset")
    image = cv2.imread(each)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, i, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('demo/dataset-result/Z05353401-bw.jpg', thresh)

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    image_copy = image.copy()
    c = max(contours, key = cv2.contourArea)

    black_canvas = np.zeros_like(img_gray)
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.drawContours(black_canvas, c, -1, 255, cv2.FILLED) # this gives a binary mask
    each = each.replace("dataset_deskewed", "dataset_filled")
    cv2.imwrite(each, black_canvas)

    #creating bounding rectangle with whome we cut out the detected countour
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),5)
    foreground = image[y:y+h,x:x+w]
    each = each.replace("dataset_filled", "dataset_contour")
    cv2.imwrite(each, foreground)

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()
#upscale
for each in arr:
    each = each.replace("dataset", "dataset_contour")
    # Read image
    image = cv2.imread(each)

    # Read the desired model
    path = r'/Users/lovro/Desktop/Image-preprocessing - testing branch/30.5/FSRCNN_Tensorflow-master/models/FSRCNN_x4.pb'
    sr.readModel(path)

    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("fsrcnn", 4)

    # Upscale the image
    result = sr.upsample(image)

    # Save the image
    each = each.replace("dataset_contour", "dataset_upscaled")
    cv2.imwrite(each, result)


# using adaptive threshold for getting best results for images with uneven lighting on whole document
for each in arr:
    each = each.replace("dataset","dataset_upscaled")
    img = cv2.imread(each, 0)

    image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,121,4)
    each = each.replace("dataset_upscaled","dataset_adaptive")
    cv2.imwrite(each, image)

#denoise
for each in arr:
    each = each.replace("dataset", "dataset_adaptive")
    img = cv2.imread(each,0)
    ret, bw = cv2.threshold(img, 128,255,cv2.THRESH_BINARY_INV)

    connectivity = 4
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 20 #threshhold value for small noisy components
    img2 = np.zeros((output.shape), np.uint8)

    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    res = cv2.bitwise_not(img2)
    each = each.replace("dataset_adaptive", "dataset_denoise")
    cv2.imwrite(each, res)


# font thinning achieved with "thin_font" function, used for improving readability of document for both human and ocr
for each in arr:
    # thinner font
    each = each.replace("dataset","dataset_denoise")
    image = cv2.imread(each)
    eroded_image = thin_font(image)
    each = each.replace("dataset_denoise", "dataset_thin")
    cv2.imwrite(each, eroded_image)

#blur
for each in arr:
    each = each.replace("dataset", "dataset_thin")
    img = cv2.imread(each, 0)
    blur = cv2.blur(img,(5,5))

    each = each.replace("dataset_thin", "dataset_final")
    cv2.imwrite(each, blur)





#here starts OCR script
dir = os.getcwd()
dir = os.path.join(dir,"dataset_final")
dir = dir.replace("\\", "/")

arr = []
arr_names = []
for filename in os.scandir(dir):
    if filename.is_file():
        arr.append(filename.path)
        arr_names.append(filename.name)
i = 0

dir = os.getcwd()
dir = os.path.join(dir,"ocr")
dir = dir.replace("\\", "/")
os.chdir(dir) #postion in folder where you want to save images

#script for iterating through images and saving results in .txt file
for each in arr:
    file_name = os.path.join(os.path.dirname(__file__), each)
    assert os.path.exists(file_name)
    img = cv2.imread(file_name, -1)
    custom_config = r'--oem 1 srp_latn+hrv --psm 6'
    pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.3.0_1/bin/tesseract'
    string = pytesseract.image_to_string(img, config=custom_config)
    string = string.lower()

    #odrediste procitanog teksta, ime_slike.txt
    txt_name = str(arr_names[i].replace(".jpg", "")) + ".txt"
    text_file = open(txt_name, "w")
    i += 1
    text_file.write(string)
    text_file.close()

    #r'/opt/homebrew/Cellar/tesseract/5.3.0_1/bin/tesseract' mac
    #r'C:\Program Files\Tesseract-OCR\tesseract.exe' win