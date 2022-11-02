import tkinter as tk
from tkinter.filedialog import askdirectory
import os
from numpy import minimum
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


root = tk.Tk()
root.withdraw()
path = askdirectory(title = 'select folder')        

fold_files = os.listdir(path)

def mmap(user):

    fig = plt.figure(figsize=(17,14.5))

    ax=plt.axes(projection=ccrs.PlateCarree())

    lons =df['LON']
    lats =df['LAT']
    z = df[user]

    cs=ax.tricontourf(lons, lats, z,  locator=ticker.MaxNLocator(150),
                        origin='lower',
                        transform = ccrs.PlateCarree(),cmap='jet',extend='both')

    ax.coastlines()

    cbar = plt.colorbar(cs)
    plt.show()

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
       with open("most_recent.txt", "w") as f1:
         for line in f:
            f1.write(line)

    df = pd.read_csv("most_recent.txt",delim_whitespace=True)
    df['Prec'][df['Prec']<0] = 0

    user = input('\t\t\t\t----------Please type what you would like to visualize----------\n\n1. AOD\n2. Precipitation\n')

    check(user)

else:
    print('error')