# import the necessary packages
from cgitb import grey
from cv2 import displayStatusBar
from imutils.contours import sort_contours
import numpy as np
import pytesseract
import argparse
import imutils
import sys
import cv2
import time
from pandas import DataFrame

areyouwindows = True
dispToggle = False
if dispToggle:
    dispConArr = {#debugging controls [wait, debug screen]
                "image":[1,1],
                "gray":[1,1],
                "blackhat":[1,1],
                "rect close":[1,1],
                "square close":[1,1],
                "dilate":[1,1],
                "rotated":[1,1],
                "cropped":[1,1],
                "bounded":[1,1]
                }
else:
    dispConArr = {#debugging controls [wait, debug screen]
            "image":[1,1],
            "gray":[0,0],
            "blackhat":[0,0],
            "rect close":[0,0],
            "square close":[0,0],
            "dilate":[0,0],
            "rotated":[0,0],
            "cropped":[0,0],
            "bounded":[1,1]
            }


#For previewing CV image at each step if needed
def displayCVimg(cvImage, name = "Output", wait = False, width = 960, height = 720, debugging = False):
    if debugging:
        cv2.imshow(name, cv2.resize(cvImage, (width, height)))
        if wait:
            cv2.waitKey(0)

#For adding bounding box for each passed text: 
#df is a dataframe, cvImg is the image which you want add the bounding box to
def boundingBox(df,cvImg, pause = False, db = True):
    #for i in range(n_boxes):
    #   if int(parsedinfoText['conf'][i]) > 60:
    for i, row in df.iterrows():
        (x, y, w, h) = (df.at[i,'left'], df.at[i,'top'], df.at[i,'width'], df.at[i, 'height'])
        boundedImg = cv2.rectangle(cvImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    displayCVimg(boundedImg,name='bounded', wait = pause, debugging=db)

#https://stackoverflow.com/questions/58657014/opencv-copy-irregular-contour-region-to-another-image-after-contour-rotation
def getContourCenter(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    else:
        return 0, 0
    return int(cx), int(cy)

def rotateContour(contour, center: tuple, angle: float):

    def cart2pol(x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return theta, rho

    def pol2cart(theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    # Translating the contour by subtracting the center with all the points
    norm = contour - [center[0], center[1]]

    # Convert the points to polar co-ordinates, add the rotation, and convert it back to Cartesian co-ordinates.
    coordinates = norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    # Convert the new polar coordinates to cartesian co-ordinates
    xs, ys = pol2cart(thetas, rhos)
    norm[:, 0, 0] = xs
    norm[:, 0, 1] = ys

    rotated = norm + [center[0], center[1]]
    rotated = rotated.astype(np.int32)

    return rotated


def halfContour(cnt):
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    print(leftmost, rightmost, topmost, bottommost)

    offset = 150
    botBound = bottommost[1]
    topBound = topmost[1]
    leftBound= leftmost[0]
    rightBound = rightmost[0]

    print(leftBound, rightBound, topBound, botBound)
    cx,cy=getContourCenter(c)
    newContour = []
    for i in cnt:
        #check if it is left line contour
        if (leftBound+offset) >= i[0][0] >= leftBound:
            if i[0][1]<cy:
                print("removedleft: ", i)
            else:
                newContour.append(i)
                print("keptleft: ", i)
        #check if it is right line contour
        elif rightBound>= i[0][0] >= (rightBound+offset):
            if i[0][1]<cy:
                print("removedright: ", i)
            else:
                newContour.append(i)
                print("keptright: ", i)
        #check if it is top line contour
        elif botBound>= i[0][1] >= (botBound+offset):
            newContour.append(i)
            print("bottom: ", i)
        elif (topBound+offset)>= i[0][1] >= topBound:
            i[0][0] = cx
            newContour.append(i)
            print("top adjusted: ", i)
        else:
            newContour.append(i)
    return np.array(newContour)
            


#Label OCR, Segmentation and auto alignment (<90degree)
"""
1. Import image and apply grayscale, blackhat and binarise
2. Perform dilation and find ontours for segmentation
3. Use cv2.minAreaRect to find angle to tilt back
4. tilt the first largest contour bounding

#DF data scraping
5. Find set name (preset/scanned list)
6. and ProdID 
"""

#for Windows path-ing - comment out for rpi
if areyouwindows:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

#https://pyimagesearch.com/2021/12/01/ocr-passports-with-opencv-and-tesseract/
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image to be OCR'd")
args = vars(ap.parse_args())

start_time = time.time()

# load the input image, convert it to grayscale, and grab its dimensions
image = cv2.imread(args["image"])
displayCVimg(image, name = "image", wait = dispConArr["image"][0], debugging=dispConArr["image"][1])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #set grayscale
(H, W) = gray.shape
displayCVimg(gray, name = "gray", wait = dispConArr["image"][0], debugging=dispConArr["gray"][1])

# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
# smooth the image using a 3x3 Gaussian blur and then apply a blackhat morpholigical operator to find dark regions on a light background
#gray = cv2.GaussianBlur(gray, (3, 3), 0)        #https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
displayCVimg(blackhat, name = "blackhat", wait = dispConArr["blackhat"][0], debugging=dispConArr["blackhat"][1])

# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
grad = np.absolute(grad)
(minVal, maxVal) = (np.min(grad), np.max(grad))
grad = (grad - minVal) / (maxVal - minVal)
grad = (grad * 255).astype("uint8")
#displayCVimg(grad, name = "gradient")
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(grad, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#De-Skew method: https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
#Apply dilate to merge text into meaningful lines/paragraphs.
# Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
kernelSkew = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10)) #(x,y)
dilate = cv2.dilate(thresh.copy(), kernelSkew, iterations=5)
displayCVimg(dilate, name = 'dilate', wait = dispConArr["dilate"][0], debugging=dispConArr["dilate"][1])

# Find all contours
contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)

#best Bounding area
c = max(contours, key = cv2.contourArea)
print(c)
#cnts = imutils.grab_contours(cnts)
(x,y,w,h) = cv2.boundingRect(c)

# draw the biggest contour (c) in green
boundedImg = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
displayCVimg(boundedImg,name='bounded', wait = False, debugging=True)

croppedimg = image[y:y + h, x:x + w]
displayCVimg(croppedimg,name='cropped', wait = dispConArr["cropped"][0], debugging=dispConArr["cropped"][1])

# Find largest contour and surround in min area box 
# angle= cv2.minAreaRect(contours[0])[-1]
minAreaRect = cv2.minAreaRect(c)
# Determine the angle. Convert it to the value that was originally used to obtain skewed image
angle = minAreaRect[-1]
if angle < -45:
    angle = (90 + angle)
elif angle >45:
    angle = (angle-90)
angle = -1*angle
# rotate the image to deskew it
(h_s, w_s) = croppedimg.shape[:2]
center = (w_s // 2, h_s // 2)
M = cv2.getRotationMatrix2D(center, -angle, 1.0)
rotated = cv2.warpAffine(croppedimg, M, (w_s, h_s),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# show the output image
print("[INFO] angle: {:.3f}".format(-angle))

cx, cy = getContourCenter(c)
rotatedContour = rotateContour(c,(cx, cy),angle)   

grayR = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)  #set grayscale
(H, W) = gray.shape
#displayCVimg(gray, name = "gray", wait = dispConArr["image"][0], debugging=dispConArr["gray"][1])

# smooth the image using a 3x3 Gaussian blur and then apply a blackhat morpholigical operator to find dark regions on a light background
#gray = cv2.GaussianBlur(gray, (3, 3), 0)        #https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
blackhatR = cv2.morphologyEx(grayR, cv2.MORPH_BLACKHAT, rectKernel)
#displayCVimg(blackhat, name = "blackhat", wait = dispConArr["blackhat"][0], debugging=dispConArr["blackhat"][1])


#displayCVimg(cvImage=bcImg, name = 'rotatedCropped',  wait = dispConArr["rotated"][0], debugging=dispConArr["rotated"][1])

#Applying OCR to segmented area
#custom_oem_psm_config = r'--oem 3 --psm 6 -l eng' #https://stackoverflow.com/questions/2363490/limit-characters-tesseract-is-looking-for
custom_oem_psm_config = r'--oem 3 --psm 6 -l eng -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz /" ' #https://stackoverflow.com/questions/2363490/limit-characters-tesseract-is-looking-for
#output to Dataframe

'''
@TIMO
This one line of code is for OCR tesseract VVV
'''
#infoText = pytesseract.image_to_data(bcImg, config = custom_oem_psm_config, output_type=pytesseract.Output.DATAFRAME)
infoText = pytesseract.image_to_data(blackhatR, config = custom_oem_psm_config, output_type=pytesseract.Output.DATAFRAME) #https://stackoverflow.com/questions/61461520/does-anyone-knows-the-meaning-of-output-of-image-to-data-image-to-osd-methods-o

print(time.time()-start_time, "seconds") #print time taken
print(infoText)

#Filter and accept those above confidence value
parsedinfoText = infoText.query('conf > 30.0')
print(parsedinfoText)
# show the info image
#cv2.imshow("info", info)
#cv2.waitKey(0)
boundingBox(df=infoText,cvImg=rotated, pause=dispConArr["bounded"][0], db=dispConArr["bounded"][1])


'''
#Windows
python OCR3.py --image images\abc4.jpg


#RPI
python OCR3.py --image model_pics/abc2.jpg
'''


