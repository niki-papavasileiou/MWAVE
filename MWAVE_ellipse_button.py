from sklearn.covariance import EllipticEnvelope
from math import radians, cos, sin, asin, sqrt 
from tkinter.filedialog import askdirectory
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from sklearn.cluster import DBSCAN
import matplotlib.ticker as ticker
import cartopy.feature as cfeature
import tkinter.scrolledtext as st
import matplotlib.pyplot as plt
from ttkthemes import ThemedTk
from sklearn.svm import SVR
import cartopy.crs as ccrs
from tkinter import ttk
from tkinter import *
import tkinter as tk
import pandas as pd
import numpy as np
import math  
import sys
import os

"""
MWAVE (METEOSAT Weather Alert and Visualization Environment)
MWARM (METEOSAT Weather Alert and Remote Monitoring)

NEED:
        me i xoris cbar.set_label(bar)                          
        color bar precipitation threshold       0-25/30 0.5-1, 1 mexri 5 ana 0.5 , 5-10 ana 1 , 10-20 ana 2 , 20-30 ana 5  // aod 0-1 ana 0.1, ana miso mexri 5 , 1-2 na 0.2 , 2-5 ana miso
        alert threshold aod/prec     
        display_ellipse an den ikanopoiite to threshold ti na bgazo                    
_____________________________________________________________________________________________________
IDEAS:
        add noaa, ecmwf data (wind speed, dir etc)[info]
        diff. alarm?
_____________________________________________________________________________________________________
PREDICTION:

    -PREC-

1h - 15min 
        Decision Tree: MSE=3.8240309196316934e-07, MAE=1.318631352024541e-06, R2=0.9999978616679438         9sec  para poli pithano overfitting
        Random Forest: MSE=1.3879367966356632e-08, MAE=3.8376719357693646e-07, R2 = 0.9999999223889711      +++++++ sec
 ->     SVM - MSE: 0.0098, MAE: 0.0987, R2 Score: 0.9453                                                    12sec Sigura oxi overfitting

        Decision Tree - Average training score: 1.0
        Decision Tree - Average test score: 0.9999559783171431
        Random Forest - Average training score: 0.9999976056839156
        Random Forest - Average test score: 0.9999402294939896
        SVM - Average training score: 0.9443328766326419
        SVM - Average test score: 0.9449091031719673

    -AOD550nm-

1h -15min 
        Decision Tree: MSE=0.05133421231952331, MAE=0.04634249191515367, R2=0.12140441966339355
        Random Forest: MSE=0.030373784298989077, MAE=0.039873654859820216, R2=0.48014644742021395
        SVM - MSE: 0.0096, MAE: 0.0972, R2 Score: 0.8600

        Decision Tree - Average training score: 1.0
        Decision Tree - Average test score: 0.0810006804529102
        Random Forest - Average training score: 0.9260606348248327
        Random Forest - Average test score: 0.4641215850731559
        SVM - Average training score: 0.8593825642352347
        SVM - Average test score: 0.8599873602796984
"""

global counter
counter = 0

def info_predict():
    global text_info, label_frame_info

    category_info = "Category: " + category +"\n"
    prediction_str = 'Short-term prediction'
    
    text_info = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
    text_info.place(x=35,y=17)
    text_info.insert(tk.INSERT, category_info)
    text_info.insert(tk.INSERT, prediction_str)
    text_info.configure(state ='disabled')

def file_comb(n_files):
    global df_comb
    pd.options.mode.chained_assignment = None  

    files_to_combine = []
    for i in range(index-n_files, index+1):
        file_path = f"{path}/{real_data[i]}"
        with open(file_path, 'r') as f:
            contents = f.read()
            time_str = real_data[i][-8:-4]
            time = int(time_str)
            lines = contents.split("\n")
            for j, line in enumerate(lines):
                if line.strip():
                    if i == index-n_files and j == 0:  # add title to new column
                        files_to_combine.append(f"{line.strip()}  time\n")
                    elif i != index-n_files and j == 0:  # skip first line for other files
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

    df_comb['Prec'] = df_comb['Prec'].clip(lower=0)
    df_comb['time'] = pd.to_datetime(df_comb['time'], format='%H%M').dt.time
    df_comb = df_comb.set_index('time')
    
    return df_comb

def alert(user):
    global label, counter

    if user == "Prec":
        threshold = 6
    else:
        threshold = 5

    if (ellipse_df[alert_var] > threshold).any():
        Label(root, text = "\t\tALERT").place(x = 30, y = 480) 
    else:
        label = tk.Label(root, text = """\t------ALERT------
                    aod prec 5 mics fks qxf dekxwhxejwha
                    """,font=('calibri', 11,'bold'))
        label.place(x=0,y=340)
        
    counter = 3

def info_ellipse():
    global text_info, label_frame_info

    category_info = "Category: " + category +"\n"
    date()
    
    text_info = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
    text_info.place(x=35,y=17)
    text_info.insert(tk.INSERT, category_info)
    text_info.insert(tk.INSERT, date_info)
    text_info.configure(state ='disabled')

def info():
    global text_info, label_frame_info

    category_info = "Category: " + category +"\n"
    date()
    
    text_info = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
    text_info.place(x=35,y=17)
    text_info.insert(tk.INSERT, category_info)
    text_info.insert(tk.INSERT, date_info)
    text_info.configure(state ='disabled')

def date():
    global date_info 

    date = str(max_data)
    year = date[0:len(date)-8]
    month = date[4:len(date)-6]
    day = date[6:len(date)-4]
    hour = date[8:len(date)-2]
    mins =date[10:len(date)-0]
    date_info = "Date: " +day +"/" +month +"/" +year +" " +hour +":" +mins  

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

    about_text = "thesis"

    text_about.insert(tk.INSERT, about_text)
    text_about.configure(state ='disabled')

def recent_file():
    global index, N, real_data, data, max_data

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
    global  category, ax, df, alert_var, count, unique_labels, dangerous_points, labels, label

    count = 2
    if counter == 3:
        label.destroy()

    plt.close()
    if user == 'AOD550nm':
        category = 'AOD550nm'
        alert_var = 'AOD550nm'
        vmin = 0
        vmax =5
        threshold = 3
        bar = 'AOD'.format(threshold)
    elif user == 'Prec':
        category = 'Precipitation'
        alert_var = 'Prec'
        vmin = 0
        vmax = 30
        threshold = 6
        bar = 'Precipitation mm/hr'.format(threshold)
            
    df = pd.read_csv("most_recent.txt",delim_whitespace=True)
    df['Prec'][df['Prec']<0] = 0

    latitudes = df["LAT"].to_numpy()
    longitudes = df["LON"].to_numpy()
    values = df[user].to_numpy()
    dangerous_points = np.column_stack((longitudes[values >= threshold], latitudes[values >= threshold]))
    if dangerous_points.size > 0:
        dbscan = DBSCAN(eps=0.5, min_samples=3).fit(dangerous_points)
        labels = dbscan.labels_
        unique_labels = set(labels)

        fig, ax = plt.subplots(figsize=(7,7), subplot_kw={'projection': ccrs.PlateCarree()})
        cs = ax.tricontourf(longitudes, latitudes, values, vmin=vmin, vmax=vmax, origin='lower',  
                            locator=ticker.MaxNLocator(150), cmap='jet', extend='both')

        for label in unique_labels:
            if label != -1:
                cluster_points = dangerous_points[labels == label]
                ee = EllipticEnvelope().fit(cluster_points)

                global height, width, angle, center
                center = ee.location_
                covariance = ee.covariance_
                height = 5*np.sqrt(covariance[1, 1])
                width = 5*np.sqrt(covariance[0, 0])
                angle = np.rad2deg(np.arccos(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])))
                ellipse = Ellipse(xy=center, width=width,
                                height=height,
                                angle=angle,
                                edgecolor='red', fc='None',lw=1.5,  transform=ccrs.PlateCarree())
                ax.add_patch(ellipse)

        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS,linestyle=':')
        cbar = fig.colorbar(cs, ax=ax, shrink=0.4)
        cbar.set_label(bar)
        plt.tight_layout()
        plt.show(block=False)
        ellipse_file()
        info_ellipse()
        cities_ellipse()
        alert(user)
    else:
        print("e")
        
def display(user):
    global  category, ax, df, count, ellipse
    
    count = 1
    if counter == 3:
        label.destroy()
    plt.close()
      
    if user == 'AOD550nm':
        category = 'AOD550nm'
        ellipse = "AOD550nm"
        vmin = 0
        vmax = 5
    elif user == 'Prec':
        category = 'Precipitation'
        ellipse = "Prec"
        vmin = 0
        vmax =20
        
    df = pd.read_csv("most_recent.txt",delim_whitespace=True)
    df['Prec'][df['Prec']<0] = 0

    fig = plt.figure(figsize=(7,6))
    ax=plt.axes(projection=ccrs.PlateCarree())

    lons =df['LON']
    lats =df['LAT']
    z = df[user]
    
    cs = ax.tricontourf(lons, lats, z, vmin=vmin, vmax=vmax ,locator=ticker.MaxNLocator(150),
                        origin='lower',
                        transform = ccrs.PlateCarree(),cmap='jet',extend='both')

    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    plt.colorbar(cs,shrink=0.5)

    text_city = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
    text_city.place(x=35,y=175)
    text_city.configure(state ='disabled')

    plt.tight_layout()
    plt.show(block=False)
    info()

def ellipse_file():
    global width, height, angle, center, og, ellipse_df, ellipse_points

    all_ellipse_data = pd.DataFrame()
    for label in unique_labels:
        if label != -1:
            cluster_points = dangerous_points[labels == label]
            ee = EllipticEnvelope().fit(cluster_points)        
            center = ee.location_
            covariance = ee.covariance_
            height = 5*np.sqrt(covariance[1, 1])
            width = 5*np.sqrt(covariance[0, 0])
            angle = np.rad2deg(np.arccos(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])))
            
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
            all_ellipse_data = pd.concat([all_ellipse_data, ellipse_df], ignore_index=True)

    with open("ellipse_data.txt",'w') as f:
        f.write(all_ellipse_data.to_string(index=False))

def cities_ellipse():
    global affected_cities, text_city, label_frame_city, cities

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
    text_city.place(x=35,y=175)
    if len(cities) == 0:
        mess = "There are no affected cities."
        text_city.insert(tk.INSERT, mess)
    else:
        text_city.insert(tk.INSERT, str(affected_cities))
    text_city.configure(state ='disabled')

def refresh():
    global max_data, N_new, N, count, real_data, index

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
            elif count ==2:   
                display_ellipse(user)

    root.after(1000, refresh)   

def predict(user2):
    global category

    plt.close()
    df_comb = file_comb(3)
    if user2=='Prec':
        vmin = 0
        vmax = 20
        category = 'Precipitation'
    else:
        vmin = 0
        vmax = 5
        category = 'AOD550nm'
    
    df = df_comb.dropna()
    df = df[['LAT', 'LON', user2]]
    X = df[['LAT', 'LON', user2]].values
    y = df[user2].values
    
    # from sklearn.ensemble import RandomForestRegressor
    # rf = RandomForestRegressor(random_state=42, n_estimators=100)
    # rf.fit(X, y)
    # y_pred = rf.predict(X)

    # from sklearn.tree import DecisionTreeRegressor
    # model = DecisionTreeRegressor(random_state=42)
    # model.fit(X, y)
    # y_pred = model.predict(X)

    svm = SVR(kernel='linear')
    svm.fit(X, y)
    y_pred = svm.predict(X)
    
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    cs = ax.tricontourf(df_comb['LON'], df_comb['LAT'], y_pred, vmin=vmin, vmax=vmax, locator=ticker.MaxNLocator(150),
                        origin='lower',
                        transform = ccrs.PlateCarree(),cmap='jet',extend='both')
    
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    plt.colorbar(cs,shrink=0.5)
    plt.tight_layout()
    plt.show(block=False)
    
    text_city = st.ScrolledText(root, width = 39, height = 8, font = ("calibri",10))
    text_city.place(x=35,y=175)
    text_city.configure(state ='disabled')
    info_predict()

def historical(user2):
    df_comb = file_comb(8)
    df_comb = df_comb[['LAT', 'LON', user2]]

    if user2=='Prec':
        vmin = 0 
        vmax = 20
    else:
        vmin = 0
        vmax = 5

    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6]) 
    
    def animate(i):
        ax.clear()
        data = df_comb.loc[df_comb.index == df_comb.index.unique()[i]]

        cs = ax.tricontourf(data["LON"], data["LAT"], data[user2], vmin = vmin,vmax = vmax ,cmap="jet", transform=ccrs.PlateCarree())
        if user2 == 'Prec':
            ax.set_title("Precipitation for {}".format(data.index[0]))
        else:
            ax.set_title("AOD for {}".format(data.index[0]))
        
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')   
        cb = fig.colorbar(cs, cax=cax, ticks=np.linspace(0, 20, 11))

    ani = animation.FuncAnimation(fig, animate, frames=len(df_comb.index.unique()), interval=500)
    ani.save("animated_plot.gif", writer="pillow")
    display_gif()

def display_gif():
    from PIL import Image, ImageTk, ImageSequence

    gif = tk.Toplevel(root)
    gif.title("Historical Data")
    pil_image = Image.open("animated_plot.gif")

    frames = []
    for frame in ImageSequence.Iterator(pil_image):
        frame = frame.convert('RGBA')
        frames.append(frame)

    image = ImageTk.PhotoImage(frames[0])
    label = tk.Label(gif, image=image)
    label.image = image
    label.grid(row=0, column=0)

    def update_image(frame_number=0):
        if label.winfo_exists():
            image = ImageTk.PhotoImage(frames[frame_number])
            label.config(image=image)
            label.image = image

            root.after(350, update_image, (frame_number + 1) % len(frames))

    root.after(0, update_image)

root = ThemedTk(theme='xpnative')
root.geometry('352x490')
root.title('MWAVE')
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
display_.add_command(label="AOD", command = lambda:display('AOD550nm'))
display_.add_command(label="Precipitation", command = lambda:display('Prec'))

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
text_city.place(x=35,y=175)
text_city.configure(state ='disabled')

label_frame_alert = ttk.LabelFrame(root, text='')
label_frame_alert.pack(expand='yes', fill='both')

check = BooleanVar(root)
checkbutton = ttk.Checkbutton(root, text='Real-time', command=lambda: refresh(),variable = check)
checkbutton.place(x=270,y=435)

predict_button_prec = ttk.Button(root, text="predict prec.", command=lambda: predict('Prec'))
predict_button_prec.place(x=95,y=435)
predict_button_aod = ttk.Button(root, text="predict AOD", command=lambda: predict('AOD550nm'))
predict_button_aod.place(x=95,y=405)

hist_button_prec = ttk.Button(root, text="prec. 2h ago", command=lambda: historical('Prec'))
hist_button_prec.place(x=10,y=435)
hist_button_aod = ttk.Button(root, text="AOD 2h ago", command=lambda: historical('AOD550nm'))
hist_button_aod.place(x=10,y=405)

ellipse_button_aod = ttk.Button(root, text="ellipse", command=lambda: display_ellipse(ellipse))
ellipse_button_aod.place(x=180,y=435)

root.config(menu=menubar)
root.protocol("WM_DELETE_WINDOW", sys.exit)
root.mainloop()
