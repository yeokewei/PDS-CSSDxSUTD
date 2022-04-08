from BomScan import BomScan
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
python runcode.py --image images\2kPaper2.png

Rpi
python runcode.py --image images/2kPaper2.jpg
'''
start_time = time.time()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image to be OCR'd") #images\abc10.jpg
args = vars(ap.parse_args())



#print(args["image"])
#print(type(args["image"]))

BOM = BomScan(path = args["image"], windows = True, dispToggle=0)

output = BOM.scan()

#BOM.path = r'images\2kPDSF2.jpg' #incase you need to change path on the fly
#output = BOM.scan()

print(output) #return dictionary with quantity

print(time.time()-start_time, "seconds") #print time taken