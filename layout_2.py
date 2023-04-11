from math import radians, cos, sin, asin, sqrt 
from sklearn.tree import DecisionTreeRegressor
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import math  
import sys
import os

global counter
counter = 0

def info_forecast():

    global text_info,label_frame_info

    category_info = "Category: " + category +"\n"
    forecast_str = 'Short- term Forecast'
    
    text_info = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
    text_info.place(x=35,y=17)
    text_info.insert(tk.INSERT, category_info)
    text_info.insert(tk.INSERT, forecast_str)
    text_info.configure(state ='disabled')

def file_comb():

    global df_comb

    new_index = [index, index-1, index-2, index-3, index-4, index-5,index-7, index-8]
    pd.options.mode.chained_assignment = None  

    files_to_combine = []
    for i in range(index-8, index+1):
        file_path = f"{path}/{real_data[i]}"
        with open(file_path, 'r') as f:
            contents = f.read()
            time_str = real_data[i][-8:-4]
            time = int(time_str)
            lines = contents.split("\n")
            for j, line in enumerate(lines):
                if line.strip():
                    if i == index-8 and j == 0:  # add title to new column
                        files_to_combine.append(f"{line.strip()}  time\n")
                    elif i != index-8 and j == 0:  # skip first line for other files
                        continue
                    else:
                        files_to_combine.append(f"{line.strip()}  {time}\n")
    with open("combined_files.txt", "w") as f:
        f.writelines(files_to_combine)
    
    df_comb = pd.read_csv('combined_files.txt', delim_whitespace=' ', dtype='unicode')

    df_comb = df_comb.astype({
    'LAT': float,
    'LON': float,
    'BT5': float,
    'BT6': float,
    'BT7': float,
    'BT9': float,
    'BT10': float,
    'Prec': float,
    'Hail': float,
    'Lgtns': float,
    'AOD550nm': float,
    'time':int
    })

    return df_comb

def texts_display():

    text_city = st.ScrolledText(label_frame_city, width = 39, height = 8, font = ("calibri",10))
    text_city.grid(column=0, row=0, ipadx=10, ipady=10) 
    text_city.configure(state ='disabled')

def alert():

    global label,counter

    if alert_var == "Prec":
        threshold = 20
    else:
        threshold = 5

    if (ellipse_df[alert_var] > threshold).any():
        Label(root, text = "\t\tALERT").place(x = 35, y = 480) 
    else:
        label = tk.Label(root, text = """\t------ALERT------
                        aod 5 mics fks qxf dekxwhxejwha
                    """,font=('calibri', 11,'bold'))
        label.place(x = 5, y = 440) 

    counter = 3

def info_ellipse():
    
    global text_info,label_frame_info

    category_info = "Category: " + category +"\n"

    area()
    date()
    
    text_info = st.ScrolledText(label_frame_info, width = 39, height = 8, font = ("calibri",10))
    text_info.grid(column=0, row=0, ipadx=10, ipady=10) 
    text_info.insert(tk.INSERT, category_info)
    text_info.insert(tk.INSERT, date_info)
    text_info.insert(tk.INSERT, area_text)
    text_info.configure(state ='disabled')

def info():
    
    global text_info,label_frame_info

    category_info = "Category: " + category +"\n"

    date()
    
    text_info = st.ScrolledText(label_frame_info, width = 39, height = 8, font = ("calibri",10))
    text_info.grid(column=0, row=0, ipadx=10, ipady=10) 
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

    global  category, ax, df, alert_var, fig, count, counter

    if counter == 3:
        label.destroy()

    count = 2
    if user == 'AOD550nm':
        category = 'AOD550nm'
        alert_var = 'AOD550nm'
        vmin = 0 
        vmax = 5
    elif user == 'Prec':
        category = 'Precipitation'
        alert_var = 'Prec'
        vmin = 0
        vmax = 20
        
    plt.close()
    df = pd.read_csv("most_recent.txt",delim_whitespace=True)
    df['Prec'][df['Prec']<0] = 0

    fig = plt.figure(figsize=(7,6))

    ax=plt.axes(projection=ccrs.PlateCarree())

    lons =df['LON']
    lats =df['LAT']
    z = df[user]

    cs = ax.tricontourf(lons, lats, z, vmin = vmin , vmax = vmax, locator=ticker.MaxNLocator(150),
                        origin='lower',
                        transform = ccrs.PlateCarree(),cmap='jet',extend='both')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    plt.colorbar(cs, orientation="horizontal",shrink=0.7)

    ellipse_file()

    # plt.show(block=False)
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=root)  
    canvas.get_tk_widget().place(x=360,y=6)
    canvas.draw()

    info_ellipse()
    cities_ellipse()
    alert()

def display(user):

    global  category, ax, df, count, fig, canvas, counter

    if counter ==3:
            label.destroy()

    if user == 'AOD550nm':
        category = 'AOD550nm'
        vmin = 0
        vmax = 5
    elif user == 'Prec':
        category = 'Precipitation'
        vmin = 0
        vmax = 20
    
    count = 1
    df = pd.read_csv("most_recent.txt",delim_whitespace=True)
    df['Prec'][df['Prec']<0] = 0

    fig = plt.figure(figsize=(7,6))

    ax=plt.axes(projection=ccrs.PlateCarree())

    lons =df['LON']
    lats =df['LAT']
    z = df[user]

    cs = ax.tricontourf(lons, lats, z,vmin = vmin, vmax =vmax,  locator=ticker.MaxNLocator(150),
                        origin='lower',
                        transform = ccrs.PlateCarree(),cmap='jet',extend='both')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    plt.colorbar(cs,orientation="horizontal",shrink=0.7)   
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=root)  
    canvas.get_tk_widget().place(x=360,y=6)
    canvas.draw()
    # plt.show(block=False)
    info()
    texts_display()

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

    text_city = st.ScrolledText(label_frame_city, width = 39, height = 8, font = ("calibri",10))
    text_city.grid(column=0, row=0, ipadx=10, ipady=10) 
    text_city.insert(tk.INSERT, str(affected_cities))
    text_city.configure(state ='disabled')

def refresh():

    global max_data, N_new, N, count, index

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
                display(user)
                count = 0
            elif count ==2:   
                display_ellipse(user)
                count = 0

        root.after(1000, refresh)  

def forecast(user2):

    global category

    df_comb = file_comb()
    df_pred = df_comb[['LAT', 'LON', user2]].dropna()
    if user2=='Prec':
        df_pred['Prec'][df_pred['Prec']<0] = 0
        vmin = 0
        vmax = 20
        category = 'Prec'
    else:
        vmin = 0
        vmax = 5
        category = 'AOD550nm'

    X = df_pred[['LAT', 'LON']].values
    y = df_pred[user2].values

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    cs = ax.tricontourf(X[:, 1], X[:, 0], y_pred, vmin=vmin, vmax=vmax, locator=ticker.MaxNLocator(150),
                        origin='lower',
                        transform = ccrs.PlateCarree(),cmap='jet',extend='both')
    
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    plt.colorbar(cs,shrink=0.5)
    plt.tight_layout()
    plt.show(block=False)
    
    info_forecast()

root = ThemedTk(theme='xpnative')
root.geometry('1080x620')
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
label_frame_info.grid(column=0, row=0, padx=15, pady=20)
text_info = st.ScrolledText(label_frame_info, width = 39, height = 8, font = ("calibri",10))
text_info.grid(column=0, row=0, ipadx=10, ipady=10)  
text_info.configure(state ='disabled')

label_frame_city = ttk.LabelFrame(root, text='Affected Cities')
label_frame_city.grid(column=0, row=1, padx=15, pady=1)
text_city = st.ScrolledText(label_frame_city, width = 39, height = 8, font = ("calibri",10))
text_city.grid(column=0, row=0, ipadx=10, ipady=10) 
text_city.configure(state ='disabled')

check = BooleanVar(root)
checkbutton = ttk.Checkbutton(root, text='real-time',variable = check, command=refresh)
checkbutton.place(x=15,y=565)

newCanvas = Canvas(root, width=700, height=570, bg='#d9d9d9')
newCanvas.place(x=358,y=5)

checkbutton.lift(newCanvas)

forecast_button_prec = ttk.Button(root, text="forecast prec.", command=lambda: forecast('Prec'))
forecast_button_prec.place(x=260,y=560)
forecast_button_aod = ttk.Button(root, text="forecast AOD", command=lambda: forecast('AOD550nm'))
forecast_button_aod.place(x=260,y=530)

root.config(menu=menubar)
root.protocol("WM_DELETE_WINDOW", sys.exit)
root.mainloop()
