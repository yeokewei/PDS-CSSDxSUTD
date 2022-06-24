#Libraries for HX711 load cell:
import RPi.GPIO as GPIO  # import GPIO
from .hx711 import HX711  # import the class HX711

#487.39 Cell A
#455.29 Cell B

#Load cell setup start
class hx711_custom:
    def __init__(self, dpin = 26, sckpin = 19, ratio = 455.38):
        GPIO.setmode(GPIO.BCM)  # set GPIO pin mode to BCM numbering
            # Create an object hx which represents your real hx711 chip
            # Required input parameters are only 'dout_pin' and 'pd_sck_pin'\
        self._weight = 0
        self.ratio = ratio
        self.hx = HX711(dout_pin=dpin, pd_sck_pin=sckpin)
            # measure tare and save the value as offset for current channel
            # and gain selected. That means channel A and gain 128
            #CHANNEL A
            #GAIN 128

        err = self.hx.zero()
        if err:
            raise ValueError('Tare is unsuccessful.')
            #checks for error in initialising HX711

        self.hx.set_scale_ratio(ratio)
        #set ratio used to convert load cell readings into grams

    @property
    def weight(self):
        return self._weight
    @weight.setter
    def weight(self,x):
        self._weight = x
        
    def zero(self):
        self.hx.zero()
        self.hx.set_scale_ratio(self.ratio)

    def takeweight(self):
        self.weight = self.hx.get_weight_mean(20)
        return self.weight

    def cleanup(self):
        #end of code
        # Cleaning up the ports that we have used in this code. It is important to do so,
        # otherwise you might have problems when you want to use the same ports for another
        # application.
        GPIO.cleanup()