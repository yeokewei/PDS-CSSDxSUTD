# OpTick Readme
OpTick is a standalone weight-sensing and label reading prototype system designed in collaboration with Changi General Hospital as a Singapore University of Technologuy & Design (SUTD) Product Design Studio project in 2021. It aims to help to reduce the workload of nurses by assisting in the accounting of sterile logistics everyday.


## Requirements
### Hardware
* RPI4
* Arducam IMX519 16MP
* 32GB SD Card
* 5kg Load Cell with HX711 Amp
* HDMI Cable + display **OR** RPI LCD touchscreen
* Mouse + Keyboard (setup & debug)
### Software
* RPI Bullseye
* python 2.7
* [libcamera](https://www.arducam.com/docs/cameras-for-raspberry-pi/raspberry-pi-libcamera-guide/)
* [pytesseract](https://pyimagesearch.com/2021/08/16/installing-tesseract-pytesseract-and-python-ocr-packages-on-your-system/)
* [OpenCV2](https://www.jeremymorgan.com/tutorials/raspberry-pi/how-to-install-opencv-raspberry-pi/)
* [imutils](https://pypi.org/project/imutils/)
* [Streamlit](https://docs.streamlit.io/library/get-started/installation#install-streamlit-on-macoslinux)

## Setup

### HX711 + Loadcell
Connect HX711 Amp to RPI as seen below (use "[pinout](https://pinout.xyz/)" in console to find your RPI BCM numbering reference)
    
| HX711 | Connection | RPI |
| :--- | :--: |  ---: |
|PWR|->|3v3 Power|
|data_pin|->|GPIO 26|
|clock_pin|->|GPIO 19|
|GND|->|Ground|

### Arducam

For this particular arducam, you can plug with the provided 15-Pin Connector ribbon cable.

## Testing
It is recommended to test that all the components attached are in working order before deploying.

### HX711 + Loadcell

To test the amp and loadcell, goto your RPI console and run:


    cd PDS-CSSDXSUTD/main/HX711_Python3
    python3 example.py


You should see that the program has no error detecting the HX711 amp. You will be prompted to calibrate the load cell with a known weight and the numbers reflected are to be in grams. If you are to see large fluctuations in the values, this can mean that your connection is loose, the HX711 is faulty or the port is faulty. Code referenced [here](https://github.com/tatobari/hx711py).

### Arducam

To test the arducam, type this code in your console:

    libcamera-still -t 0

You should see a new window popup, showing a live feed from your connected arducam. More info [here](https://www.arducam.com/docs/cameras-for-raspberry-pi/raspberry-pi-libcamera-guide/)


## Deployment code
After preparing the requirements listed above, navigate your working directory to /main on your RPI in console and run

    streamlit run UImain.py




