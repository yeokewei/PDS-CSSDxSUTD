import streamlit as st
from PIL import Image
import os

from re import A
import numpy as np
import pandas as pd 
import time
from datetime import datetime
import pytz

#import custom classes
from BOM_Scanner.BomScan import BomScan #BOM list scanning
from Label_Checker.LabelOCR import LabelOCR #label scanning with Arducam 519
from HX711_Python3.hx711_custom import hx711_custom as HX #for HX711 with loadcell

##--------------------------------------------------------------------------------------------------------##
#button formating
m = st.markdown("""
<style>
div.stButton > button:first-child {
    color: black;
    height: 3em;
    width: 12em;
    font-family:sans-serif;
    font-size:20px ;
    background-color:#A4C2F4;
    ;
    font-weight: bold;
    border-radius:20px;
   
    font-weight: semibold;
    margin: auto;
    display: block;
    padding: 2px;
}

div.stButton > button:hover {
	background-color:#1B4079;
    border-color:#1B4079;
    color:white;
}

div.stButton > button:active {
	position:relative;
	top:3px;
    colour:F0F8FF;
    
}

</style>""", unsafe_allow_html=True)
###-------------------------------------------------------------------------------------------------------##
#metric markdown
label=""" 
<style>
.css-1ht1j8u {
    overflow-wrap: normal;
    text-overflow: ellipsis;
    width: 100%;
    overflow: hidden;
    font-size: 30px;
    white-space: nowrap;
    font-family: "Source Sans Pro", sans-serif;
    line-height: normal; 
    <style>
    """

st.markdown(label,unsafe_allow_html=True)

value=""" 
<style>
.css-1ht1j8u {
    overflow-wrap: normal;
    text-align: center;
    text-overflow: ellipsis;
    width: 100%;
    overflow: hidden;
    font-size: 50px;
    white-space: nowrap;
    font-family: "Source Sans Pro", sans-serif;
    line-height: normal; 
    <style>
    """

st.markdown(label,unsafe_allow_html=True)

delta=""" 
<style>
.css-1ht1j8u {
    overflow-wrap: normal;
    ;
    text-overflow: ellipsis;
    width: 100%;
    overflow: hidden;
    font-size: 30px;
    white-space: nowrap;
    font-family: "Source Sans Pro", sans-serif;
    line-height: normal; 
    <style>
    """
st.markdown(delta,unsafe_allow_html=True)
##--------------------------------------------------------------------------------------------------------##
#table markdown
hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)
##--------------------------------------------------------------------------------------------------------##
#setting fonts to be bigger
def header(url):
     st.markdown(f'<p style="background-color:transparent ;color:#1B4079;font-size:30px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
def header2(url):
     st.markdown(f'<p style="background-color:transparent ;color:#1B4079;font-size:23px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
def blacksmall(url):
     st.markdown(f'<p style="background-color:transparent ;color:black;font-size:20px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
def step(url):
    st.markdown(f'''
    <p style="
    background-color:transparent ;
    color:#1B4079;
    font-size:20px;
    font-weight:bold;
    border-radius:2%;
    ">{url}</p>''', unsafe_allow_html=True)
def note(url):
    st.markdown(f'''
    <p style="
    background-color:transparent ;
    color:#1B4079;
    font-size:23px;
    text-align: center;
    font-style: italic;
    font-weight:italized;
    border-radius:2%;
    ">{url}</p>''', unsafe_allow_html=True)

def Bomquantity(url):
    st.markdown(f'''
    <p style="background-color:transparent ;
    color:black;
    font-size:25px;
    border-radius:2%;
    ">{url}</p>''', unsafe_allow_html=True)

def quantity(url):
    st.markdown(f'''
    <p style="background-color:transparent ;
    color:#1B4079;
    font-size:25px;
    border-radius:2%;
    ">{url}</p>''', unsafe_allow_html=True)
##--------------------------------------------------------------------------------------------------------##
#image paths
image =Image.open(r'assets/begin.png')
image2=Image.open(r'assets/start.png')
image_confirm1=Image.open(r'assets/list.png')
image_confirm2=Image.open(r'assets/indiv.png')

##--------------------------------------------------------------------------------------------------------##
#session_states

#threshold
if "threshold" not in st.session_state: 
    st.session_state["threshold"] = 20

#Init first page state
if "page" not in st.session_state: 
    st.session_state["page"] = 'begin_page'

#BOM class object
if "BOMDict" not in st.session_state:
    bompath = r"assets/bom.png"
    BOM = BomScan(path = bompath, windows = False, dispToggle=0)
    st.session_state["BOMDict"] = BOM
#LabelOCR class object
if "LABELDf" not in st.session_state:
    labelpath = r"assets/label.png"
    LABEL = LabelOCR(path = labelpath, windows = False, dispToggle=0)
    st.session_state["LABELDf"]=LABEL


#Declare flag for BOM scanning (to scan or not to scan)
if 'scanah' not in st.session_state:
    st.session_state['scanah'] = (True,False)
    
def scanornot(yes = False, rescan = False):
    if yes and not rescan:
        st.session_state['scanah'] = (True,False)
    else:
        st.session_state['scanah'] = (False,True)
        
def initclass():
    bompath = r"assets/bom.png"
    #BOM Class
    BOM = BomScan(path = bompath, windows = False, dispToggle=0)
    st.session_state["BOMDict"] = BOM
    #LABEL Class
    labelpath = r"assets/label.png"
    LABEL = LabelOCR(path = labelpath, windows = False, dispToggle=0)
    st.session_state["LABELDf"]=LABEL
    
    st.session_state['scanah'] = (True,False)
        
        
##--------------------------------------------------------------------------------------------------------##
#RPI functions

#Function to take photo with arducam on RPI
def takePhoto(output = 0, width = 2560 ,height= 1920, af = 0, delayms = 1500, nopreview = 1):
    #output: 0 bom, 1 labelhigh, 2 labellow
    #focus: 0 not in use 1200 for bom, 1550 labelhigh 1300 labellow
    focusval = [1200,1550,1370]
    path = ["assets/bom.png","assets/label.png","assets/label2.png"]
    if af:
        focusstring = ""
        libcamstill_string = f"libcamera-still -t {delayms} --rotation 180 --autofocus {af} --height {height} --width {width} -n {nopreview} -o {path[output]}"
    else: #preset focus
        focusstring = f"v4l2-ctl -c focus_absolute={focusval[output]} -d /dev/v4l-subdev1"
        libcamstill_string = f"libcamera-still --rotation 180 --immediate --height {height} --width {width} -n {nopreview} -o {path[output]}"
    if focusstring != "":
        os.system(focusstring)
    os.system(libcamstill_string)

#Loadcell function
def returnWeight():
    if 'LOADCELL' not in st.session_state:
        loadCell = HX()
        st.session_state['LOADCELL'] = loadCell
    else:
        loadCell = st.session_state['LOADCELL']
    weight = loadCell.takeweight()
    print("Loadcell:",weight)
    return int(weight)

#Scan functions
def scanBOM():
    with st.spinner('Scanning BOM...'):
        #print("trying to scanbom")
        takePhoto(output = 0)
        BOMDict = st.session_state['BOMDict']
        results = BOMDict.scan() 
        st.session_state['BOMDict'] = BOMDict
    return results #either a dictionary of current scanned items or None
    
def scanLABEL(weight):
    LABELDf = st.session_state['LABELDf']
    results = LABELDf.scan(weight) #either a dictionary of current scanned items or None
    st.session_state['LABELDf'] = LABELDf
    return results

##--------------------------------------------------------------------------------------------------------##
# counting of number of list scanned
if 'count' not in st.session_state:
    st.session_state['count'] = 0
def increment_counter():
    st.session_state['count'] += 1

##--------------------------------------------------------------------------------------------------------##
# pages definition()
def begin_page():
    initclass()
    st.session_state["page"]='begin_page' 
def start_page():
    st.session_state["page"]='start_page'   

def listscan_page():
    st.session_state["page"]='listscan_page'  
def confirm_finishpage():
    st.session_state["page"]='confirm_finishpage'
    
    
#OCR individual 
def scan_page():
    st.session_state["page"]='scan_page'
def ind_scanf():
    st.session_state["page"]='ind_scan'
    st.session_state['scanah'] = (True,False)
def ind_scan():
    st.session_state["page"]='ind_scan'
def secconfirm_finishpage():
    st.session_state["page"]='secconfirm_finishpage'
def overview_page():
    st.session_state["page"]='overview_page'
    
##(page indicator)-----------------------------------------------------------------------------------------##
st.write("Current Page:", st.session_state["page"])


###----(page 0)-------------------------------------------------------------------------------------------##
if st.session_state["page"] == 'begin_page': 
    st.image(image)
    if st.button('Begin', on_click = start_page):
        st.spinner('Wait for it......')
##--------------------------------------------------------------------------------------------------------##



#-(start scan page)---------------------------------------------------------------------------------------##
if st.session_state["page"] == 'start_page':
    st.markdown("<h1 style='text-align:centerialised;color:#black;font-size: 15px;background-color:transparent;height:3.5em;'>Step 1:To activate system</h1>", unsafe_allow_html=True)
    st.image(image2)
    note('Note: Place A4 list under the camera before clicking the start button')
    if st.button("Scan page(s)",on_click=listscan_page):
        st.spinner('Wait for it......')


##(listscan page)----------------------------------------------------------------------------------------##
if st.session_state['page']=='listscan_page':
    step("Step 2: Process of Scanning the list")
    placeholder=st.empty()
    placeholder1=st.empty()
    #using markdown to prevent misalignment of buttons
    col1,col2 =st.columns(2) #for buttons
    #for output BOM
    (yes, rescan) = st.session_state['scanah']
    #Start Scanning of BOM
    if not rescan:
        output = scanBOM() #store scanned information
        print(output)
        if output != None:
            placeholder1.success('Successful Scan')
            increment_counter()
            #Show current count with metric
            BOMDict = st.session_state['BOMDict'] #get global BOM object
            for key,val in BOMDict.BOMDict.items(): #loop through each item
                if output[key] == 0: #If no increments
                    colour = "off"
                else:
                    colour = "normal" 
                st.metric(key,str(val),output[key], delta_color = colour)
            print(BOMDict.BOMDict)
            with col1:
                #st.markdown("<h1 style='text-align: center; color: black;font-size: 15px;'>**Note: Select Continue to scan more list</h1>",unsafe_allow_html=True)
                col1.button('Next List', on_click=listscan_page)
                with placeholder.container():
                    st.write('List Scanned = ', st.session_state['count'])
        else:
            placeholder1.error("Scan error, please realign and scan.")
            BOMDict = st.session_state['BOMDict']
            if (BOMDict.BOMDict['Podiatry Dressing Set Vascular'] == 0 and BOMDict.BOMDict['Podiatry Dressing Set Clinic 1'] == 0) or output == None: #if dict is empty show error bar only
                pass
            else: #show previous counts
                for key,val in BOMDict.BOMDict.items(): #loop through each item
                    if output[key] == 0: #If no increments
                        colour = "off"
                    else:
                        colour = "normal" 
                st.metric(key,str(val),output[key], delta_color = colour)
            col1.button("Retry",on_click= listscan_page) #Show retry button
            
    else: #case where user returns from next screen, only show current count
        col1.button('Continue Scan', on_click=listscan_page)
        with placeholder.container():
            st.write('List Scanned = ', st.session_state['count'])
        #Show current count with metric
        BOMDict = st.session_state['BOMDict'] #get global BOM object
        for key,val in BOMDict.BOMDict.items():
            st.metric(key,str(val),"0", delta_color = "off")
        scanornot(True,False)

    with col2: 
        if col2.button('Finish Scan', on_click=confirm_finishpage):
            st.spinner('Wait for it......')
            
###-(confirm finish page 1)--------------------------------------------------------------------------------##
if st.session_state['page']=='confirm_finishpage':
    scanornot(False,True) #(takephoto,rescan or not?)
    
    #https://discuss.streamlit.io/t/how-do-i-align-st-title/1668/11
    
    placeholder=st.empty()
    with placeholder.container():
        st.image(image_confirm1)
       
    col1,col2 =st.columns(2)
    col1.button('Back', on_click=listscan_page)
    col2.button('Continue', on_click=ind_scanf)
##(individual scan page)------------------------------------------------------------------------------------##
if st.session_state['page']=='ind_scan':
    step("Step 4: Scanning of individual sets")
    header("Realtime Data")
    
    
    (yes, rescan) = st.session_state['scanah']
    threshold = st.session_state["threshold"]
    LABELDf = st.session_state["LABELDf"]
    flag = "None"
    name = ""
    ID = ""
    weight=""
    
    #col1,col2 = st.columns([1,1])
    weightholder = st.empty()
    nameholder = st.empty()
    idholder = st.empty()
    placeholder = st.empty()
    
            
    if not rescan:
        weight = returnWeight()
        print('load cell started:',weight)
        
        
        ###loop through and get weight
        with weightholder:
            
            while not((344-threshold) <= weight <= (344+threshold)) and not((1037-threshold) <= weight <= (1037+threshold)):
                weight = returnWeight()
                #st.write(f"{weight} current weight")
                string_weight = "Weight    : "+ str(weight)+"g"
                weightholder.markdown(f'## {string_weight}')
            
        if (344-threshold) <= weight <= (344+threshold): #scan for PVS
            takePhoto(output=2)
            LABELDf.path = r"assets/label2.png"
            dataDict = LABELDf.scan(weight)
            ID = dataDict["ProdID"]
            name = dataDict["ProdName"]
            flag = "PDS"
            
            
        elif (1037-threshold) <= weight <= (1037+threshold): #scan for PDS
            takePhoto(output=1)
            LABELDf.path = r"assets/label.png"
            dataDict = LABELDf.scan(weight)
            ID = dataDict["ProdID"]
            name = dataDict["ProdName"]
            flag = "PVS"
        
        if ID == None or ID == "" or name == None or name =="": #if not able to scan for any parts
            placeholder.error("Scan error, please realign and scan.") #display error
            LABELDf.df.drop(index=LABELDf.df.index[-1],axis=0,inplace=True) #remove last entry which has NONE
            st.session_state['LAEBLDf'] = LABELDf #save the updated DF
            flag= "None"
            
        elif LABELDf.df.duplicated(subset=['ProdID']).any(): #if ID is duplicated
            LABELDf.df.drop_duplicates(subset=['ProdID'],inplace=True) #remove the ID that are duplicated
            placeholder.error(f"Product ID:  {ID}  is duplicated") #printout ID that was duplicated
            print(LABELDf.df)
            st.session_state['LAEBLDf'] = LABELDf #save the updated DF
            flag = "Dup"
            
        else:
            #display success
            string_weight = "Weight    : "+ str(weight)+"g"
            weightholder.markdown(f'## {string_weight}')
            
            placeholder.success("Successful Scan")
            string_name = "Name: "+ str(name)
            nameholder.markdown(f'## {string_name}')
            
            string_ID = "ID    : "+ str(ID)
            idholder.markdown(f'## {string_ID}')

        st.markdown('----------------')
        
        c1,c2 = st.columns(2)
        if flag == "None":
            c1.button('Retry Scan', on_click=ind_scanf)
        else:
            c1.button('Next Scan', on_click=ind_scanf)
        c2.button('Finish Scan', on_click=secconfirm_finishpage)
        col_1,col_2,col_3 =st.columns([2,1,1])
        
        BOM = st.session_state["BOMDict"]
        prodlist = BOM.BOMDict.keys()
        
        with col_1:
            
            header2("Name")
            for prodname in prodlist:
                blacksmall(str(prodname))
            
        with col_2:
            
            header2("Current Count")
            counts = LABELDf.df.ProdName.value_counts()
            for prodname in prodlist:
                try:
                    quantity(counts[prodname])
                except:
                    quantity(0)
                    
        with col_3:
            
            header2("List Count")

            for prodname in prodlist:
                Bomquantity(BOM.BOMDict[prodname])
        
        if not(LABELDf.df.empty): #if no label item has been scanned
            checked = st.checkbox('View all sets')
            scanornot(False,False)
            if checked:
                st.subheader('View all sets below')
                listnames=['Podiatry Dressing Set Clinic 1', 'Podiatry Dressing Set Vascular']
                selectedsets=st.selectbox('Select your sets',listnames)
                if selectedsets == listnames[0]:
                    printdf = LABELDf.df.query('ProdName == "Podiatry Dressing Set Clinic 1"')
                    st.table(printdf)
                else:
                    printdf = LABELDf.df.query('ProdName == "Podiatry Dressing Set Vascular"')
                    st.table(printdf)  
        
    else:

        
        c1,c2 = st.columns(2)

        c1.button('Next Scan', on_click=ind_scanf)
        c2.button('Finish Scan', on_click=secconfirm_finishpage)
        
        col_1,col_2,col_3 =st.columns([2,1,1])
        
        BOM = st.session_state["BOMDict"]
        prodlist = BOM.BOMDict.keys()
        
        with col_1:
            
            header2("Name")
            for prodname in prodlist:
                blacksmall(str(prodname))
            
        with col_2:
            
            header2("Current Count")
            counts = LABELDf.df.ProdName.value_counts()
            for prodname in prodlist:
                try:
                    quantity(counts[prodname])
                except:
                    quantity(0)
                    
        with col_3:
            
            header2("List Count")

            for prodname in prodlist:
                Bomquantity(BOM.BOMDict[prodname])
                
        
        if not(LABELDf.df.empty): #if no label item has been scanned
            checked = st.checkbox('View all sets')
            scanornot(False,False)
            if checked:
                st.subheader('View all sets below')
                listnames=['Podiatry Dressing Set Clinic 1', 'Podiatry Dressing Set Vascular']
                selectedsets=st.selectbox('Select your sets',listnames)
                if selectedsets == listnames[0]:
                    printdf = LABELDf.df.query('ProdName == "Podiatry Dressing Set Clinic 1"')
                    st.table(printdf)

                else:
                    printdf = LABELDf.df.query('ProdName == "Podiatry Dressing Set Vascular"')
                    st.table(printdf)

        
#####(second finish page)----------------------------------------------------------------------------------------------------
if st.session_state['page']=='secconfirm_finishpage':
    scanornot(False,True) #(takephoto,rescan or not?)
    st.image(image_confirm2)
    col1,col2 =st.columns(2)

    col1.button('Back', on_click=ind_scan)
    col2.button('Finish Scan', on_click=overview_page)
if st.session_state['page']=='overview_page':
    st.markdown("<h1 style='color:#black;font-size: 15px;height:3.5em;'>Step 6: Overview of the sets scanned </h1>", unsafe_allow_html=True)
    st.markdown("<h1 style=' color:#191970;font-size: 25px ;height:2.5em;;'>Overview </h1>", unsafe_allow_html=True)
    st.header=("Overview")
    st.markdown('--------')
    
    col_1,col_2,col_3 =st.columns([2,1,1])
    
    BOM = st.session_state["BOMDict"]
    LABELDf = st.session_state["LABELDf"]
    prodlist = BOM.BOMDict.keys()
    
    with col_1:
        
        header2("Name")
        for prodname in prodlist:
            blacksmall(str(prodname))
        
    with col_2:
        
        header2("Current Count")
        counts = LABELDf.df.ProdName.value_counts()
        for prodname in prodlist:
            try:
                quantity(counts[prodname])
            except:
                quantity(0)
    with col_3:
        
        header2("List Count")

        for prodname in prodlist:
            Bomquantity(BOM.BOMDict[prodname])
            
    st.button('Return Home', on_click=begin_page)

