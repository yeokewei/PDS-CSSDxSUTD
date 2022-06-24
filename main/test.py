#https://docs.python.org/3/reference/import.html#package-relative-imports
#Importing BOM_Scanner, Label_Checker and HX711

from BOM_Scanner.BomScan import BomScan
from Label_Checker.LabelOCR import LabelOCR

import argparse
import time

'''
For debugging purposes:
initialise BomScan(path = r'path_here'), requires to declare path to image location
Call method> .scan() to OCR the image at the path and return a dictionary count of the current scan
ie {'Podiatry Dressing Set Vascular': 18, 'Podiatry Dressing Set Clinic 1': 0}

Total scan count stored within .BOMlist (can be called)

Cmd line
Windows
python main.py --image images\2kPaper2.png
python main.py --image BOM_Scanner\images\2kPaper2.png

Rpi
python main.py --image images/2kPaper2.jpg
python main.py --image BOM_Scanner/images/2kPaper2.png
'''
start_time = time.time()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image to be OCR'd") #images\abc10.jpg
args = vars(ap.parse_args())

print(args["image"])

BOM = BomScan(path = args["image"], windows = True, dispToggle=0)

output = BOM.scan()

print(output) #return dictionary with quantity

print(time.time()-start_time, "seconds") #print time taken

start_time = time.time()

locr = LabelOCR(path = r"Label_Checker\images\2k10_1.jpg", windows = True, dispToggle=0)

output2 = locr.scan(100)

print(time.time()-start_time, "seconds") #print time taken

print(output2)