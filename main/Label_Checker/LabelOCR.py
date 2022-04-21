# import the necessary packages
#from matplotlib.font_manager import _Weight
import numpy as np
from pandas import DataFrame,concat
import pytesseract
import imutils
import cv2
import time
import re

#import sys
#import argparse

'''
Initialise LabelOCR(path = r'path_here'), requires to declare path to image location
Call method> .start(insert_weight_here) to scan image from path and passes in weight value
Returns dict := {'Weight': weight, 'ProdName': ProdName, 'ProdID': ProdID}

.df -> returns stored df of the class
.path = r'images\2kPDSF2.jpg'

This CV+OCR programme autoaligns labelks (<=40 degrees) at 2560x1920 resolution (assuming no light reflections)
Scans and rips out Label Name (ProdName) and Unique Serial number (ProdID)



#Windows
python LabelOCR.py --image images\abc4.jpg

#RPI
python LabelOCR.py --image model_pics/2k10_1.jpg
'''
class LabelOCR:
    def __init__(self, path = '', windows = False, dispToggle = 0): #initilise 

        self.setNames = {1:['Podiatry', 'Dressing', 'Set', 'Vascular'], 2:['Podiatry', 'Dressing', 'Set', 'Clinic', '1']}
        self._df = DataFrame(columns = ["Weight", "ProdName", "ProdID"])
        self._path = path
        #dispToggle: 0 0ff, 1 Partial, 2 All
        if dispToggle == 0: #Off Debugging mode
            self.dispConArr = {#debugging controls [wait, debug screen]
                "image":[0,0],
                "gray":[0,0],
                "blackhat":[0,0],
                "rect close":[0,0],
                "square close":[0,0],
                "dilate":[0,0],
                "rotated":[0,0],
                "Biggest Bound":[0,0],
                "cropped":[0,0],
                "cropped bottom":[0,0],
                "Final Image":[0,0]
                }
        elif dispToggle == 1: #On Specific debugging mode
            self.dispConArr = {#debugging controls [wait, debug screen]
                    "image":[0,1],
                    "gray":[0,0],
                    "blackhat":[0,1],
                    "rect close":[0,0],
                    "square close":[0,0],
                    "dilate":[0,1],
                    "rotated":[0,1],
                    "Biggest Bound":[0,1],
                    "cropped":[0,1],
                    "cropped bottom":[0,1],
                    "Final Image":[1,1]
                    }
        elif dispToggle == 2:#Full debugging mode
            self.dispConArr = {#debugging controls [wait, debug screen]
                        "image":[1,1],
                        "gray":[1,1],
                        "blackhat":[1,1],
                        "rect close":[1,1],
                        "square close":[1,1],
                        "dilate":[1,1],
                        "rotated":[1,1],
                        "Biggest Bound":[1,1],
                        "cropped":[1,1],
                        "cropped bottom":[1,1],
                        "bounded":[1,1]
                        }
        self.path = path


        #for Windows path-ing - comment out for rpi
        if windows:
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

    @property
    def path(self):
        return self._path
    @path.setter
    def path(self,x):
        self._path = x

    @property
    def df(self):
        return self._df
    @df.setter
    def df(self,x):
        self._df = x

    #For previewing CV image at each step if needed
    def displayCVimg(self, cvImage, name = "Output", wait = False, width = 960, height = 720, debugging = False):
        if debugging:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            #cv2.imshow(name, cv2.resize(cvImage, (width, height)))
            cv2.imshow(name, cvImage)
            if wait:
                cv2.waitKey(0)

    #For adding bounding box for each passed text: 
    #df is a dataframe, cvImg is the image which you want add the bounding box to
    def boundingBox(self, df,cvImg, pause = False, db = True):
        #for i in range(n_boxes):
        #   if int(parsedinfoText['conf'][i]) > 60:
        for i, row in df.iterrows():
            (x, y, w, h) = (df.at[i,'left'], df.at[i,'top'], df.at[i,'width'], df.at[i, 'height'])
            boundedImg = cv2.rectangle(cvImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self.displayCVimg(boundedImg,name='Final Image', wait=pause, debugging=db, height=360)


    #https://stackoverflow.com/questions/38665277/crop-half-of-an-image-in-opencv
    def crop_bottom_half(self,image): #cuts image into half
        cropped_img = image[int(image.shape[0]/2):int(image.shape[0]), 0:image.shape[1]]
        return cropped_img

    
    """
    #Label OCR, Segmentation and auto alignment (<=40degree in either direction)
    1. Import image and apply grayscale, blackhat and binarise
    2. Perform dilation and find contours for segmentation
    3. Use cv2.minAreaRect to find angle to tilt back
    4. tilt the first largest contour bounding

    #DF data scraping
    5. Find set name (preset/scanned list)
    6. and ProdID 
    """
    
    def scan(self, weight):
        #https://pyimagesearch.com/2021/12/01/ocr-passports-with-opencv-and-tesseract/
        # construct the argument parser and parse the arguments
        '''
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True,help="path to input image to be OCR'd")
        args = vars(ap.parse_args())
        '''
        start_time = time.time()

        # load the input image, convert it to grayscale, and grab its dimensions
        #print("currentpath: ",self.path)
        image = cv2.imread(self.path)
        self.displayCVimg(image, name = "image", wait = self.dispConArr["image"][0], debugging= self.dispConArr["image"][1])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #set grayscale
        self.displayCVimg(gray, name = "gray", wait = self.dispConArr["image"][0], debugging= self.dispConArr["gray"][1])

        # initialize a rectangular and square structuring kernel
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        #sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        # smooth the image using a 3x3 Gaussian blur and then apply a blackhat morpholigical operator to find dark regions on a light background
        #gray = cv2.GaussianBlur(gray, (3, 3), 0)        #https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        self.displayCVimg(blackhat, name= "blackhat", wait= self.dispConArr["blackhat"][0], debugging= self.dispConArr["blackhat"][1])

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
        kernelSkew = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10)) #(x,y) do not put too large or your crop area will be too big
        dilate = cv2.dilate(thresh.copy(), kernelSkew, iterations=5)
        self.displayCVimg(dilate, name = 'dilate', wait = self.dispConArr["dilate"][0], debugging= self.dispConArr["dilate"][1])

        # Find all contours
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)

        #best Bounding area
        c = max(contours, key = cv2.contourArea)
        #print(c)
        #cnts = imutils.grab_contours(cnts)

        #(x,y,w,h) = cv2.boundingRect(c)
        minAreaRect = cv2.minAreaRect(c)
        box = cv2.boxPoints(minAreaRect)
        box = np.int0(box)
        #print(box)
        (x,y,w,h) = cv2.boundingRect(box)

        # draw the biggest contour (c) in green
        boundedImg = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
        self.displayCVimg(boundedImg,name='Biggest Bound', wait = self.dispConArr["Biggest Bound"][0], debugging= self.dispConArr["Biggest Bound"][1])

        croppedimg = gray[y:y + h, x:x + w]
        #self.displayCVimg(croppedimg,name='cropped', wait = self.dispConArr["cropped"][0], debugging= self.dispConArr["cropped"][1])
        self.displayCVimg(croppedimg,name='cropped', wait = self.dispConArr["cropped"][0], debugging= self.dispConArr["cropped"][1])


        #Deskew angle
        #if landscape or portrait
        landscape = True
        #angle2 = self.getOrientation(c)
        #print(angle2)
        
        angle = minAreaRect[-1]
        print("Angle,W,H:", angle, w, h)
        if w<h and abs(h-w) >50: #for those labels that are portrait
            landscape = False
            if 45>angle>0:
                angle = (-90+angle)
            elif 0>angle>-45:
                angle = (-90-angle)
            else:
                pass
        else:
            if angle < -45:
                angle = (90 + angle)
            elif angle >45:
                angle = -(90 - angle)
            else:
                pass
                #angle = (90 - angle)

        # rotate the image to deskew it
        if angle >= 0.05 or angle<=-0.05:
            if landscape:
                (h_s, w_s) = croppedimg.shape[:2]
                center = (w_s // 2, h_s // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(croppedimg, M, (w_s, h_s),
                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            else:
                rotated = imutils.rotate_bound(croppedimg, angle)
        
            self.displayCVimg(rotated, name = "rotated", wait= self.dispConArr["rotated"][0], debugging= self.dispConArr["rotated"][1], height=360)
        
        else:
            rotated = croppedimg
            
        print("[INFO] angle: {:.3f}".format(angle))

        '''
        grayR = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)  #set grayscale
        (H, W) = gray.shape
        #displayCVimg(gray, name = "gray", wait = dispConArr["image"][0], debugging=dispConArr["gray"][1])
        '''
        croppedBottom = self.crop_bottom_half(rotated)
        self.displayCVimg(croppedBottom, name= "cropped bottom", wait= self.dispConArr["cropped bottom"][0], debugging= self.dispConArr["cropped bottom"][1], width= 960, height=360)

        #blackhatR = cv2.morphologyEx(croppedR, cv2.MORPH_BLACKHAT, rectKernel)
        '''
        kernel = np.ones((1,1),np.uint8)
        dilated_img = cv2.dilate(blackhatR, kernel, iterations = 5)
        '''
        #Applying OCR to segmented area 
        #https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/
        #PSM 6. Assume a Single Uniform Block of Text
        #custom_oem_psm_config = r'--oem 3 --psm 6 -l eng -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz /" ' #https://stackoverflow.com/questions/2363490/limit-characters-tesseract-is-looking-for

        #Sparse Text: Find as Much Text as Possible in No Particular Order 
        #custom_oem_psm_config = r'--oem 3 --psm 11 -l eng -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz /" '

        #PSM 3. Fully Automatic Page Segmentation, But No OSD
        custom_oem_psm_config = r'--oem 3 --psm 3 -l eng -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz /" ' #https://stackoverflow.com/questions/2363490/limit-characters-tesseract-is-looking-for

        #Output OCR data into DF
        #infoText = pytesseract.image_to_data(bcImg, config = custom_oem_psm_config, output_type=pytesseract.Output.DATAFRAME)
        infoText = pytesseract.image_to_data(croppedBottom, config = custom_oem_psm_config, output_type=pytesseract.Output.DATAFRAME) #https://stackoverflow.com/questions/61461520/does-anyone-knows-the-meaning-of-output-of-image-to-data-image-to-osd-methods-o

        #print(type(infoText))
        print(time.time()-start_time, "seconds") #print time taken
        #print(infoText)
        infoText['text'] = infoText['text'].astype('str') #change 'text' column to string type for better manipulation

        infoText.to_csv(r'C:\Users\User\OneDrive - Singapore University of Technology and Design\MODS\T4\60.003 Product Design Studio\Projs\RPI\3OCR\Code\Label_Checker\out.csv', encoding='utf-8', index=False)

        #show bounding boxes on final image (to check how image is OCRed)
        self.boundingBox(df=infoText,cvImg=croppedBottom, pause= self.dispConArr["Final Image"][0], db= self.dispConArr["Final Image"][1])

        #Filter and accept those above confidence value
        parsedinfoText = infoText.query('conf > 70.0')
        #print(parsedinfoText)
        ProdID = None
        ProdName = None
        packedRow = None

        #print(infoText.dtypes)
        #Extract ProdId with regex
        try: #extracting by checking main df for 14 digit char (because sometimes OCR will give 0% conf despite finding the ProdID)
            packedRow = parsedinfoText.query('text == "Packed"').index.item()
            PIDrow = infoText['text'].str.match(r'(^[0-9]{14})')
            ProdID = infoText['text'][PIDrow].values[0]
            #print(ProdID, "(0 conf)")
        except:
            try: #extracting ProdID by finding 'Packed' and looking backwards (hardcoded method)
                packedRow = parsedinfoText.query('text == "Packed"').index.item()
                #print(packedRow)
                temp = infoText._get_value(packedRow-1,'text').replace(" ", "")
                #print("temp type:", temp)
                if len(temp)<14:
                    ProdID = infoText._get_value(packedRow-2,'text') + infoText._get_value(packedRow-1,'text')
                else:
                    ProdID = temp
                ProdID.replace(" ", "")
                #print(len(ProdID))
            except:
                print('ProdID not found')
                ProdID = None
        '''
        ProdNameLine = parsedinfoText.query('text == @setNames[1][0]').index.item()
        print(packedRow , ProdNameLine)

        tempName = infoText[(infoText['line_num'] == ProdNameLine) & (infoText['word_num'] == ProdWordLine) & (infoText.index>packedRow)]['text'].values[0]
        print(tempName)
        '''
        #print('Prod Id type: ',type(ProdID))
        if re.match(r'^[0-9]{14}$',ProdID) == None: #ProdID does not meet the requirement of 14 numerical digits
            IDlen = len(ProdID)
            print(ProdID, f'of length {IDlen} does not match')
            ProdID = None

        try:#Find 'Packed' row and offset 2 lines down
            ProdNameRow = parsedinfoText.query('text == @self.setNames[1][0]').index.item()
            tempName = None
            dfName = ""
            while  True:
                #tempName = infoText[(infoText['line_num'] == ProdNameLine) & (infoText['word_num'] == ProdWordLine) & (infoText.index>packedRow+10)]['text'].values[0]
                tempName = infoText.iloc[ProdNameRow, infoText.columns.get_loc('text')]
                print(tempName)
                if tempName == '/':
                    break
                else:
                    dfName += tempName + " "
                    ProdNameRow +=1
            dfName = dfName[:-1]
            errorName = False
            #print(dfName)
        except:
            errorName = True
            print("Product Name can't be found")
        finally:
            if not(errorName):
                ProdName = dfName
            else:
                ProdName = None

        dataDict = {'Weight': weight, 'ProdName': ProdName, 'ProdID': ProdID}
        dataDf = DataFrame.from_records(dataDict, index=[0])
        self.df = concat([self.df,dataDf] , ignore_index=True)
        print(self.df)
        print(dataDict)
        return dataDict







