###Initial RPI Setup
Use RPI imager to install in fresh fat32 sd card (more than 8gb space at least)
 For iso/disc image/ choose bullseye for latest 64bit system
 Change to legacy or 32bit version if your current rpi is not compatible

Once installed, set your user name and password. Enable SSH and connect to wifi (use the code below to configure SSH)

>sudo raspi-config

Install xRDP on RPI(for remote access to RPI via windows)

>sudo apt-get update
>sudo apt-get install xrdp


Find your rpi ip (look under inet for wlan0)

>ifconfig

Use remote desktop connection and type in the ip address
 when promptedwith user and pw, use the same one that you have created at the beginning
 If not use, pi and raspberry
