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

    center = [16.581,38.634]
    width = 5.                                   #horizontal axis
    height = 2.501828848                         #vertical axis 
    angle = -50.                                  #anti-clockwise rotation
    e = patches.Ellipse(xy=center, width=width, height=height, angle = 180+angle,  edgecolor='r', facecolor='none')
    ax.add_patch(e)

    Area = 3.142 * width * height
    #print(Area)  

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
    test = yo[np.where(rad_cc <= 1.)[0]] 

    og = sys.stdout

    with open("ell_points.txt",'w') as f:
        sys.stdout = f 
        np.set_printoptions(threshold=np.inf)
        print(test)
        sys.stdout = og

    plt.show()

    #new = df.filter(['LON','LAT','AOD550nm'], axis=1)   
    #newnew = new['AOD550nm'].idxmax()
    #maxrow = np.array(new.loc[newnew])
    #center = maxrow[0:2]

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

    print(real_data[index])

    with open(f'{path}/{real_data[index]}', 'r') as f:                                    
     with open("most_recent.txt", "w") as f1:
        for line in f:
            f1.write(line)

    df = pd.read_csv("most_recent.txt",delim_whitespace=True)
    df['Prec'][df['Prec']<0] = 0

    user = input('\t\t\t\t----------Please type what you would like to visualize----------\n\n1. AOD\n2. Precipitation\n')

    check(user)
    

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
