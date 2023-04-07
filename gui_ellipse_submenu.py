from math import radians, cos, sin, asin, sqrt 
from tkinter.filedialog import askdirectory
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import cartopy.feature as cfeature
import tkinter.scrolledtext as st
import matplotlib.pyplot as plt
from ttkthemes import ThemedTk
import cartopy.crs as ccrs
from tkinter import ttk
from tkinter import *
import tkinter as tk
import pandas as pd
import numpy as np
import math  
import sys
import os

def alert():

    global text_alert

    if alert_var == "Prec":
        threshold = 20
    else:
        threshold = 5

    if (ellipse_df[alert_var] > threshold).any():
        msg = "alert"
    else:
        msg = "no alert"

    text_alert = st.ScrolledText(root, width = 39, height = 3, font = ("calibri",10))
    text_alert.place(x=35,y=340)
    text_alert.insert(tk.INSERT, msg)
    text_alert.configure(state ='disabled')

def info_ellipse():
    
    global text_info,label_frame_info

    category_info = "Category: " + category +"\n"

    area()
    date()
    
    text_info = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
    text_info.place(x=35,y=17)
    text_info.insert(tk.INSERT, category_info)
    text_info.insert(tk.INSERT, date_info)
    text_info.insert(tk.INSERT, area_text)
    text_info.configure(state ='disabled')

def info():
    
    global text_info,label_frame_info

    category_info = "Category: " + category +"\n"

    date()
    
    text_info = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
    text_info.place(x=35,y=17)
    text_info.insert(tk.INSERT, category_info)
    text_info.insert(tk.INSERT, date_info)
    text_info.configure(state ='disabled')

def area():
 
    global  Area, area_text

    lat1_w = radians(center_y)
    lat2_w = radians(center_y+width/2)

    lng1_w = radians(center_x)
    lng2_w = radians(center_x+width/2)

    lat1_h = radians(center_y)
    lat2_h = radians(center_y+height/2)

    lng1_h = radians(center_x)
    lng2_h = radians(center_x+height/2)

    dlat_w = lat2_w - lat1_w
    dlng_w = lng2_w - lng1_w

    dlat_h = lat2_h - lat1_h
    dlng_h = lng2_h - lng1_h

    a_w = sin(dlat_w / 2)**2 + cos(lat1_w) * cos(lat2_w) * sin(dlng_w / 2)**2
    a_h = sin(dlat_h / 2)**2 + cos(lat1_h) * cos(lat2_h) * sin(dlng_h / 2)**2

    c_h = 2 * asin(sqrt(a_h)) 
    c_w = 2 * asin(sqrt(a_w)) 
    
    r = 6371
    w =c_w * r
    h =c_h * r

    Area = 3.142 * w/2 * h/2 
    area_text = "Area: " +str(round(Area, 3)) + "  km^2\n"

def date():
    
    global date_info 

    date = str(max_data)
    year = date[0:len(date)-8]
    month = date[4:len(date)-6]
    day = date[6:len(date)-4]
    hour = date[8:len(date)-2]
    mins =date[10:len(date)-0]
    date_info = "Date: " +day +"/" +month +"/" +year +" " +hour +":" +mins  +"\n"  

def about_window():

    about_win = Toplevel(root)
    about_win.title("About...")
    about_win.geometry("500x600")
    about_win.resizable(0,0)

    info_photo = tk.PhotoImage(file = 'info.png')
    about_win.wm_iconphoto(False, info_photo)

    label_frame_about = ttk.LabelFrame(about_win, text='About...')
    label_frame_about.pack(expand='yes', fill='both')
    text_about = st.ScrolledText(about_win, width = 60, height = 36, font = ("calibri",10))
    text_about.place(x=35,y=15)

    about_text = "ptixiaki"

    text_about.insert(tk.INSERT, about_text)
    text_about.configure(state ='disabled')

def recent_file():

    global index, N,real_data, data, max_data

    N = len(fold_files)
    data = []                                           
    real_data = []
    for file in fold_files:
        real_data.append(file)
        data.append(int(file[7:len(file)-4]))               

    i = 0
    for file in fold_files:
        if max(data) == int(file[7:len(file)-4]):
            index = i
        i+=1

    with open(f'{path}/{real_data[index]}', 'r') as f:                                    
        with open("most_recent.txt", "w") as f1:
            for line in f:
                f1.write(line)

    max_data = max(data)
    
def open_file():

    global fold_files, path
    path = askdirectory(title = 'select folder')        
    fold_files = os.listdir(path)
    recent_file()

def display_ellipse(user):

    global clear_button, category, ax, df, alert_var

    
    plt.close()
    if user == 'AOD550nm':
        category = 'AOD550nm'
        alert_var = 'AOD550nm'
    elif user == 'Prec':
        category = 'Precipitation'
        alert_var = 'Prec'
        

    df = pd.read_csv("most_recent.txt",delim_whitespace=True)
    df['Prec'][df['Prec']<0] = 0

    fig = plt.figure(figsize=(7,6))

    ax=plt.axes(projection=ccrs.PlateCarree())

    lons =df['LON']
    lats =df['LAT']
    z = df[user]

    cs = ax.tricontourf(lons, lats, z,  locator=ticker.MaxNLocator(150),
                        origin='lower',
                        transform = ccrs.PlateCarree(),cmap='jet',extend='both')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    plt.colorbar(cs)

    ellipse_file()
    plt.tight_layout()
    plt.show(block=False)

    info_ellipse()
    cities_ellipse()
    alert()

def display(user):
    global clear_button, category, ax, df, count
    
    plt.close()
      
    if user == 'AOD550nm':
        category = 'AOD550nm'
    elif user == 'Prec':
        category = 'Precipitation'
        
    count = 1
    df = pd.read_csv("most_recent.txt",delim_whitespace=True)
    df['Prec'][df['Prec']<0] = 0

    fig = plt.figure(figsize=(7,6))

    ax=plt.axes(projection=ccrs.PlateCarree())

    lons =df['LON']
    lats =df['LAT']
    z = df[user]

    cs = ax.tricontourf(lons, lats, z,  locator=ticker.MaxNLocator(150),
                        origin='lower',
                        transform = ccrs.PlateCarree(),cmap='jet',extend='both')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    plt.colorbar(cs)

    text_alert = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
    text_alert.place(x=35,y=340)
    text_alert.configure(state ='disabled')
    text_city = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
    text_city.place(x=35,y=180)
    text_city.configure(state ='disabled')

    plt.tight_layout()
    plt.show(block=False)
    info()

def ellipse_file():

    global width,height, angle, center_x, center_y, og, ellipse_df,ellipse_points

    df_ellipse_file = pd.read_csv("ellipse_test.txt", sep = ',')

    width = df_ellipse_file['width'].iloc[-1]
    height = df_ellipse_file['height'].iloc[-1]
    angle = df_ellipse_file['angle'].iloc[-1]
    center_x = df_ellipse_file['centre_x'].iloc[-1]
    center_y = df_ellipse_file['centre_y'].iloc[-1]

    center = [center_x,center_y]
    ellipse = patches.Ellipse(xy=center, width=width, height=height, angle = 180+angle,  edgecolor='r', facecolor='none')
    ax.add_patch(ellipse)

    filter = df.filter(['LON','LAT'], axis=1)   
    xy = np.array(filter)  

    cosine = np.cos(np.radians(180. - angle))
    sine = np.sin(np.radians(180. - angle)) 
    xc = xy[:,0] - center[0]
    yc = xy[:,1] - center[1]  
    xct = cosine * xc - sine * yc
    yct = sine * xc + cosine * yc  
    rad_cc = (xct**2/(width/2.)**2) + (yct**2/(height/2.)**2)   
    points = np.array(xy)

    ellipse_points = points[np.where(rad_cc <= 1.)[0]] 

    ellipse_df = df[df['LON'].isin(ellipse_points[:,0]) & df['LAT'].isin(ellipse_points[:,1])]

    og = sys.stdout

    with open("ellipse_data.txt",'w') as f:
        sys.stdout = f 
        np.set_printoptions(threshold=np.inf)
        print(ellipse_df.to_string())
        sys.stdout = og

    #ellipse_file_len = df_ellipse_file.shape[0]

def cities_ellipse():

    global affected_cities,text_city, label_frame_city, cities

    cities_df = pd.read_csv("cities.csv", sep = ',')

    d = 3
    def truncate(f, n):
        return math.floor(f * 10 ** n) / 10 ** n
    
    city_lat = cities_df['lat'].astype(float).apply(lambda number: truncate(number, d))
    city_lng = cities_df['lng'].astype(float).apply(lambda number: truncate(number, d))
    
    cities = cities_df[city_lng.isin(ellipse_points[:,0]) & city_lat.isin(ellipse_points[:,1])]
    pd.options.mode.chained_assignment = None
    cities['city_info'] = cities['city'] + ', ' + cities['admin_name'] + ', ' + cities['country']
    affected_cities = cities['city_info'].to_string(index=False)

    text_city = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
    text_city.place(x=35,y=180)
    text_city.insert(tk.INSERT, str(affected_cities))
    text_city.configure(state ='disabled')

def refresh():

    global max_data, N_new, N, count

    if check.get():

        if category == 'AOD550nm':
            user = 'AOD550nm'       
        elif category == 'Precipitation':
            user = 'Prec'

        fold_files_new =os.listdir(path)
        N_new = len(fold_files_new)              

        if (N_new != N):

            data = []                                           
            real_data = []
            for file in fold_files_new:
                real_data.append(file)
                data.append(int(file[7:len(file)-4]))               

            i = 0
            for file in fold_files_new:
                if max(data) == int(file[7:len(file)-4]):
                    index = i
                i+=1

            with open(f'{path}/{real_data[index]}', 'r') as f:                                    
                with open("most_recent.txt", "w") as f1:
                    for line in f:
                        f1.write(line)    
                                                                                        
            N = N_new
            max_data = max(data)

            if count == 1:
                plt.close()
                display(user)
                count = 0
            else:
                
                display_ellipse(user)

        root.after(1000, refresh)   

root = ThemedTk(theme='xpnative')
root.geometry('420x500')
root.title('Meteosat Observer')
root.resizable(0,0)

photo = tk.PhotoImage(file = 'icon.png')
root.wm_iconphoto(False, photo)

menubar = Menu(root)

file = Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=file)
file.add_command(label='Open...', command = open_file)
file.add_separator()
file.add_command(label="Exit", command=root.quit)

display_ = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Display', menu=display_)

sub_menu_aod = Menu(display_, tearoff= 0)
sub_menu_aod.add_command(label="AOD", command = lambda:display('AOD550nm'))
sub_menu_aod.add_command(label="AOD Ellipse", command = lambda:display_ellipse('AOD550nm'))

sub_menu_prec = Menu(display_, tearoff= 0)
sub_menu_prec.add_command(label="Precipitation", command = lambda:display('Prec'))
sub_menu_prec.add_command(label="Precipitation Ellipse", command = lambda:display_ellipse('Prec'))

display_.add_cascade(label = 'AOD', menu=sub_menu_aod)
display_.add_cascade(label='Precipitation', menu=sub_menu_prec)

help_ = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Help', menu=help_)
help_.add_command(label='About...', command = about_window)

label_frame_info = ttk.LabelFrame(root, text='Info')
label_frame_info.pack(expand='yes', fill='both')
text_info = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
text_info.place(x=35,y=17)
text_info.configure(state ='disabled')

label_frame_city = ttk.LabelFrame(root, text='Affected Cities')
label_frame_city.pack(expand='yes', fill='both')
text_city = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
text_city.place(x=35,y=180)
text_city.configure(state ='disabled')

label_frame_alert = ttk.LabelFrame(root, text='ALERT')
label_frame_alert.pack(expand='yes', fill='both')
text_alert = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
text_alert.place(x=35,y=340)
text_alert.configure(state ='disabled')

check = BooleanVar(root)
checkbutton = ttk.Checkbutton(root, text='real-time', command=lambda: refresh(),variable = check)
checkbutton.place(x=335,y=15)

root.config(menu=menubar)
root.mainloop()
