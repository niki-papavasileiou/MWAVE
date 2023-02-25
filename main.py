import tkinter as tk
from tkinter.filedialog import askdirectory
import os
import pandas as pd
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
import numpy as np
import sys


root = tk.Tk()
root.withdraw()
path = askdirectory(title = 'select folder')        
fold_files = os.listdir(path)

def mmap(user):

    fig = plt.figure(figsize=(8,8))

    ax=plt.axes(projection=ccrs.PlateCarree())

    lons =df['LON']
    lats =df['LAT']
    z = df[user]

    cs = ax.tricontourf(lons, lats, z,  locator=ticker.MaxNLocator(150),
                        origin='lower',
                        transform = ccrs.PlateCarree(),cmap='jet',extend='both')

    ax.coastlines()
    cbar = plt.colorbar(cs)

    center_x = float(input("Enter centre's latitude (DD): \n"))
    center_y = float(input("Enter centre's longitude (DD): \n"))
    width = float(input("Enter horizontal axis: \n"))
    height = float(input("Enter vertical axis: \n"))
    angle = float(input("Enter angle: \n"))

    center = [center_x,center_y]
    e = patches.Ellipse(xy=center, width=width, height=height, angle = 180+angle,  edgecolor='r', facecolor='none')
    ax.add_patch(e)
    plt.show()

    Area = 3.142 * width * height 

    filter = df.filter(['LON','LAT'], axis=1)   
    xy = np.array(filter)  

    cosine = np.cos(np.radians(180. - angle))
    sine = np.sin(np.radians(180. - angle))
    
    xc = xy[:,0] - center[0]
    yc = xy[:,1] - center[1]
    
    xct = cosine * xc - sine * yc
    yct = sine * xc + cosine * yc  

    rad_cc = (xct**2/(width/2.)**2) + (yct**2/(height/2.)**2)   
    
    yo = np.array(xy)
    global ellipse_points
    ellipse_points = yo[np.where(rad_cc <= 1.)[0]] 

    global ellipse_df
    ellipse_df = df[df['LON'].isin(ellipse_points[:,0]) & df['LAT'].isin(ellipse_points[:,1])]
    
    global og
    og = sys.stdout

    with open("ellipse_data.txt",'w') as f:
        sys.stdout = f 
        np.set_printoptions(threshold=np.inf)
        print(ellipse_df.to_string())
        sys.stdout = og

    print("\n\nInfo \n\n")
    Area = 3.142 * width * height 
    print("area:",Area, "km^2\n")

    cities_df = pd.read_csv("gr.csv", sep = ',')
    cities = cities_df[cities_df['lng'].isin(ellipse_points[:,0]) & cities_df['lat'].isin(ellipse_points[:,1])]
    cities['city_info'] = cities['city'] + ', ' + cities['admin_name'] + ', ' + cities['country']
    print(cities['city_info'].to_string(index=False))

def check(user):
   
    if user in ['aod','AOD', 'Aod', 'AoD','1']:
        mmap('AOD550nm')
        
        
    elif user in ['prec' ,'pREC','prEC', 'PREC' , 'Prec','2','Precipitation' ,'precipitation','PRECIPITATION']:
        mmap("Prec")

    else:
        print('error')

q = input('\t\t\t\t----------Would you like to open the most recent file? Y/N----------\n\n')
  

if q in ['Y','y', 'YES','yes']:

    data = []                                           
    real_data = []
    for file in fold_files:

        real_data.append(file)
        data.append(int(file[7:len(file)-4]))               
        #print(max(data))                                    

    i = 0
    for file in fold_files:
        if max(data) == int(file[7:len(file)-4]):
            index = i
        i+=1

    with open(f'{path}/{real_data[index]}', 'r') as f:                                    
     with open("most_recent.txt", "w") as f1:
        for line in f:
            f1.write(line)

    df = pd.read_csv("most_recent.txt",delim_whitespace=True)
    df['Prec'][df['Prec']<0] = 0

    user = input('\t\t\t\t----------Please type what you would like to visualize----------\n\n1. AOD\n2. Precipitation\n')

    check(user)
    
    new_index = [index, index-1, index-2,index-3, index-4, index-5,index-7,index-8]
    pd.options.mode.chained_assignment = None  

    f2 = open("2hfiles.txt", "w")
    for z in new_index:
        with open(f'{path}/{real_data[z]}', 'r'):
            sys.stdout = f2
            ellipse_df['file'] = np.array(real_data[z])
            np.set_printoptions(threshold=np.inf)
            print(ellipse_df.to_string())
            sys.stdout = og
    

elif q in ['n', 'N','NO','no','No']:

    input0 = input('\t\t\t\t ---------- Type a date (YYYYmmddHHMM) ---------- \n\n') 
    
    data = []                                           
    real_data = []
    for file in fold_files:
        
        real_data.append(file)
        data.append(int(file[7:len(file)-4]))    

    i = 0
    for input0 in file:
        index=i

    with open(f'{path}/{real_data[index]}', 'r') as f:                                    
       with open("date.txt", "w") as f1:
         for line in f:
            f1.write(line)

    df = pd.read_csv("date.txt",delim_whitespace=True)
    df['Prec'][df['Prec']<0] = 0

    user = input('\t\t\t\t----------Please type what you would like to visualize----------\n\n1. AOD\n2. Precipitation\n')

    check(user)

else:
    print('error')
