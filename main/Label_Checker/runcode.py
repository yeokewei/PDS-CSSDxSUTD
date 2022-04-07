#https://stackoverflow.com/questions/4142151/how-to-import-the-class-within-the-same-directory-or-sub-directory
from LabelOCR import LabelOCR
import argparse

'''
For debugging purposes:
initialise LabelOCR(path = r'path_here'), requires to declare path to image location
Call method> .start(insert_weight_here) to scan image from path and passes in weight value
Returns dict := {'Weight': weight, 'ProdName': ProdName, 'ProdID': ProdID}

Cmd line
Windows
python runcode.py --image images\abc4.jpg

Rpi
python runcode.py --image images/1920PNSF3.jpg
'''


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image to be OCR'd") #images\abc10.jpg
args = vars(ap.parse_args())

#print(args["image"])
#print(type(args["image"]))

locr = LabelOCR(path = args["image"], windows = True, dispToggle=1)

output = locr.scan(100)

locr.path = r'images\2kPDSF2.jpg' #incase you need to change path on the fly

output = locr.scan(200)

print(output)
print(locr.df) #if you want to access df
