#Libraries for HX711 load cell:
import RPi.GPIO as GPIO  # import GPIO
from hx711 import HX711  # import the class HX711
#from hx711 import outliers_filter

#Load cell setup start

GPIO.setmode(GPIO.BCM)  # set GPIO pin mode to BCM numbering
    # Create an object hx which represents your real hx711 chip
    # Required input parameters are only 'dout_pin' and 'pd_sck_pin'

hx = HX711(dout_pin=26, pd_sck_pin=19)
    # measure tare and save the value as offset for current channel
    # and gain selected. That means channel A and gain 128
    #CHANNEL A
    #GAIN 128

err = hx.zero()
if err:
    raise ValueError('Tare is unsuccessful.')
    #checks for error in initialising HX711

hx.set_scale_ratio(487.39)
#set ratio used to convert load cell readings into grams
print(hx.get_weight_mean(20), 'g') #prints the weight in grams. Loop this 

while True:
    print(hx.get_weight_mean(20), 'g')


#end of code
# Cleaning up the ports that we have used in this code. It is important to do so,
# otherwise you might have problems when you want to use the same ports for another
# application.
GPIO.cleanup()
