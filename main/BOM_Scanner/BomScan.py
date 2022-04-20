# import the necessary packages

#from matplotlib.pyplot import gray
import pytesseract
#import argparse
import imutils
import cv2
#import time
import os
import numpy as np
#import re
#import pandas

'''
#Windows
python BOMScan.py --image images\2kpaper2.jpg

#RPI
python BOMScan.py --image model_pics/paper2.jpg
'''

''''
Requires:
- Path to image
- windows bool
- Debug screen toggle (optional)

1. Scan image from path
2. Bound name portion of the image and attempt to Straighten image
3. Apply OCR to bounded-rotated image (outputs DataFrame)
4. Parse through DF and count PVS and PDS, updates dictionary count
5. returns dictionary of updated count
'''
class BomScan:
	def __init__(self, path = '', windows = False, dispToggle = 0):
		self.setNames = {1:{'Podiatry', 'Dressing', 'Set', 'Vascular'}, 2:{'Podiatry', 'Dressing', 'Set', 'Clinic', '1'}}
		self._path = path
		self.pagewords = []
		self.BOMDict = {'Podiatry Dressing Set Vascular': 0, 'Podiatry Dressing Set Clinic 1': 0}
		self.debug = dispToggle

		if windows:
			pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

		if dispToggle == 0: #Off Debugging mode
			self.dispConArr = {#debugging controls [wait, debug screen]
				"gray":[0,0],
				"blackhat":[0,0],
				"rect close":[0,0],
				"square close":[0,0],
				"dilate":[0,0],
				"rotated":[0,0],
				"Biggest Bound":[0,0],
				"cropped":[0,0],
				"rectbounds":[0,0],
				"crop":[0,0],
				"bounded":[0,0]
				}
		elif dispToggle == 1: #On Specific debugging mode
			self.dispConArr = {#debugging controls [wait, debug screen]
					"gray":[0,1],
					"blackhat":[0,1],
					"rect close":[0,1],
					"square close":[0,1],
					"dilate":[0,1],
					"rotated":[0,1],
					"Biggest Bound":[0,1],
					"cropped":[0,1],
					"rectbounds":[0,1],
					"crop":[0,1],
					"bounded":[1,1]
					}
		elif dispToggle == 2:#Full debugging mode
			self.dispConArr = {#debugging controls [wait, debug screen]
					"gray":[1,1],
					"blackhat":[1,1],
					"rect close":[1,1],
					"square close":[1,1],
					"dilate":[1,1],
					"rotated":[1,1],
					"Biggest Bound":[1,1],
					"cropped":[1,1],
					"rectbounds":[1,1],
					"crop":[1,1],
					"bounded":[1,1]
						}

	def displayCVimg(self, cvImage, name = "Output", wait = False, width = 960, height = 720, debugging = False):
		if debugging:
			cv2.namedWindow(name, cv2.WINDOW_NORMAL)
			#cv2.imshow(name, cv2.resize(cvImage, (width, height)))
			cv2.imshow(name, cvImage)
			if wait:
				cv2.waitKey(0)


	def boundingBox(self, df,cvImg, pause = False, db = True):
		#for i in range(n_boxes):
		#   if int(parsedinfoText['conf'][i]) > 60:
		for i, row in df.iterrows():
			(x, y, w, h) = (df.at[i,'left'], df.at[i,'top'], df.at[i,'width'], df.at[i, 'height'])
			boundedImg = cv2.rectangle(cvImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
		self.displayCVimg(boundedImg,name='Final Image', wait=pause, debugging=db, height=360)
    
	@property
	def path(self):
		return self._path
	@path.setter
	def path(self,x):
		self._path = x

	def scan(self):

		try:
            
			tempDict = {'Podiatry Dressing Set Vascular': 0, 'Podiatry Dressing Set Clinic 1': 0}
			#start_time = time.time()
			# load the input image from disk, resize it, and compute the ratio
			# of the *new* width to the *old* width
			image = cv2.imread(self.path)
			#https://stackoverflow.com/questions/55119504/is-it-possible-to-check-orientation-of-an-image-before-passing-it-through-pytess
			#re.search('(?<=Orientation confidence: )\d+', newdata).group(0)

			#image = orig.copy()

			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #set grayscale
			self.displayCVimg(gray, name = "gray", wait = self.dispConArr["gray"][0], debugging= self.dispConArr["gray"][1])

			# initialize a rectangular and square structuring kernel
			rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
			#sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
			# smooth the image using a 3x3 Gaussian blur and then apply a blackhat morpholigical operator to find dark regions on a light background
			#gray = cv2.GaussianBlur(gray, (3, 3), 0)        #https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
			blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
			#self.displayCVimg(blackhat, name= "blackhat", wait= self.dispConArr["blackhat"][0], debugging= self.dispConArr["blackhat"][1])

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
			
			angle = minAreaRect[-1]
			print("Angle,W,H:", angle, w, h)

			if angle < -45:
				angle = (90 + angle)
			elif angle >45:
				angle = -(90 - angle)
			else:
				pass

			# rotate the image to deskew it
			rotated = imutils.rotate_bound(croppedimg, -angle)
			print("[INFO] angle: {:.3f}".format(angle))
			self.displayCVimg(rotated, name = "rotated", wait= self.dispConArr["rotated"][0], debugging= self.dispConArr["rotated"][1], height=360)

			#PSM 3. Fully Automatic Page Segmentation, But No OSD
			custom_oem_psm_config = r'--oem 3 --psm 3 -l eng -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz /" ' #https://stackoverflow.com/questions/2363490/limit-characters-tesseract-is-looking-for

			#Output OCR data into DF
			#infoText = pytesseract.image_to_data(bcImg, config = custom_oem_psm_config, output_type=pytesseract.Output.DATAFRAME)
			BOMText = pytesseract.image_to_data(rotated, config = custom_oem_psm_config, output_type=pytesseract.Output.DATAFRAME) #https://stackoverflow.com/questions/61461520/does-anyone-knows-the-meaning-of-output-of-image-to-data-image-to-osd-methods-o

			#Remove low confidence elements
			parsedBOMText = BOMText.query('conf > 75.0')
			#parsedBOMText.dropna(inplace=True)
			parsedBOMText.reset_index(drop=True, inplace=True)
			#print(parsedBOMText)

			
			#For exporting to csv in windows (debugging purposes)
			#export df to csv
			'''
			cwd = os.getcwd() #get current working dir
			outputCSV = cwd+"\out.csv"
			BOMText.to_csv(outputCSV, encoding='utf-8', index=True)
			
			outputCSV2 = cwd+"\out2.csv"
			parsedBOMText.to_csv(outputCSV2, encoding='utf-8', index=True)
			'''
			self.boundingBox(df=parsedBOMText,cvImg=rotated, pause= self.dispConArr["bounded"][0], db= self.dispConArr["bounded"][1])

			podiatryRows = parsedBOMText.query('text == "Podiatry"').index
			#print(podiatryRows)
			#Loops through rows that starts with Podiatry and parse through to make full name
			for row in podiatryRows:
				print('current row: %s'%(row))
				currentLine = parsedBOMText._get_value(row,'line_num')
				currentPar= parsedBOMText._get_value(row,'par_num')
				print('current line: %s'%(currentLine))
				tempLine = currentLine
				tempPar = currentPar
				stringArr = []
				stringArr.append(parsedBOMText._get_value(row,'text'))
				print(stringArr)
				tempRow = row+1
				quantity = parsedBOMText._get_value(row-2,'text')
				
				while currentLine == tempLine and currentPar == tempPar:
					stringArr.append(parsedBOMText._get_value(tempRow,'text').replace(" ", ""))
					tempRow +=1
					tempLine = parsedBOMText._get_value(tempRow,'line_num')
					tempPar = parsedBOMText._get_value(tempRow,'par_num')
				#self.pagewords.append({'quantity':quantity,'string': stringArr,'row':row})
				print(stringArr)
				self.pagewords.append({'string': stringArr,'row':row})
			#print(self.pagewords)

			#Identify, count and update main count
			countPVS, countPDS = 0, 0
			for item in self.pagewords: #set(x) & set(y) returns intersections between the lists x & y and returns a dict of those elements that matches
				if set(item['string']) & set(['Podiatry', 'Dressing', 'Set', 'Vascular']) == {'Podiatry', 'Dressing', 'Set', 'Vascular'}:
					try:
						countPVS+= 1
					except:
						print('Possible Data scan error: ',item['string'], 'row: ', item['row'])
						
				elif set(item['string']) & set(['Podiatry', 'Dressing', 'Set', 'Clinic']) == {'Podiatry', 'Dressing', 'Set', 'Clinic'}:
					try:#Podiatry Nail Surgery Set Clinic 1 ?
						countPDS+= 1
					except:
						print('Possible Data scan error: ',item['string'], 'row: ', item['row'])
				else:
					if self.debug >0:
						print(item['string'])
					else:	
						pass
			if (countPVS == 0 and countPDS == 0):
				return None #able to scan but no PVS or PDS
			if countPVS > 0:
				tempDict['Podiatry Dressing Set Vascular'] = countPVS
				self.BOMDict['Podiatry Dressing Set Vascular'] += countPVS
			if countPDS > 0:
				tempDict['Podiatry Dressing Set Clinic 1'] = countPDS
				self.BOMDict['Podiatry Dressing Set Clinic 1'] += countPDS
			return tempDict

		except:
			print(f"Error {image}, please rescan")
			return None
		finally:
			self.pagewords.clear()