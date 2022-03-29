########################################################################################################################
# iCorrVision-2D Correlation Module                                                                                    #
# iCorrVision team:     João Carlos Andrade de Deus Filho,  M.Sc. (PGMEC/UFF¹)      <joaocadf@id.uff.br>               #
#                       Prof. Luiz Carlos da Silva Nunes,   D.Sc.  (PGMEC/UFF¹)     <luizcsn@id.uff.br>                #
#                       Prof. José Manuel Cardoso Xavier,   P.hD.  (FCT/NOVA²)      <jmc.xavier@fct.unl.pt>            #
#                                                                                                                      #
#   1. Department of Mechanical Engineering | PGMEC - Universidade Federal Fluminense (UFF)                            #
#     Campus da Praia Vermelha | Niterói | Rio de Janeiro | Brazil                                                     #
#   2. NOVA SCHOOL OF SCIENCE AND TECHNOLOGY | FCT NOVA - Universidade NOVA de Lisboa (NOVA)                           #
#     Campus de Caparica | Caparica | Portugal                                                                         #
#                                                                                                                      #
# Date: 28-03-2022                                                                                                     #
########################################################################################################################

'''
    iCorrVision-2D Correlation Module
    Copyright (C) 2022 iCorrVision team

    This file is part of the iCorrVision-2D software.

    The iCorrVision-2D Correlation Module is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    The iCorrVision-2D Correlation Module is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

V = 'v1.04.22' # Version

########################################################################################################################
# Modules
########################################################################################################################
import tkinter as tk; from tkinter import *
import os
from math import pi
import subprocess
from tkinter import filedialog
from numpy import genfromtxt
import numpy as np; from threading import Thread; import glob
import cv2 as cv; from PIL import Image, ImageTk; import time
from tkinter import messagebox; from PIL import Image;
import datetime; from shapely.geometry import Point; from scipy import interpolate
from shapely.geometry.polygon import Polygon; from scipy.interpolate import interp2d
import matplotlib; import ntpath
from scipy import ndimage
from matplotlib import pyplot as plt
import ctypes
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
matplotlib.use('Agg', force=True)

########################################################################################################################
# Open user guide
########################################################################################################################
def openguide(CurrentDir):
    subprocess.Popen([CurrentDir + '\static\iCorrVision-2D.pdf'], shell=True)

########################################################################################################################
# Global variables declaration for parallel computing (x0, y0, u0, v0, x1, y1, u1 and v1)
########################################################################################################################
process_x0_mem = None
process_x0_shape = None
process_y0_mem = None
process_y0_shape = None
process_u0_mem = None
process_u0_shape = None
process_v0_mem = None
process_v0_shape = None

########################################################################################################################
# Line drawing using mouse event functions
########################################################################################################################
def drawLineMouse(event,x,y,flags,param):
    global point_x1,point_y1,pressed,points, subsetImage

    if event==cv.EVENT_LBUTTONDOWN:

        pressed=True
        point_x1,point_y1=x,y
        points.append((x, y))

    elif event==cv.EVENT_MOUSEMOVE:

        if pressed==True:

            cv.line(subsetImage,(point_x1,point_y1),(x,y),color=(255,0,0),thickness=2)
            point_x1,point_y1=x,y
            points.append((x, y))

    elif event==cv.EVENT_LBUTTONUP:

        pressed=False
        cv.line(subsetImage,(point_x1,point_y1),(x,y),color=(255,0,0),thickness=2)

########################################################################################################################
# Freehand cut region using drawLineMouse function
########################################################################################################################
def freehandCut(name, img):
    global point_x1,point_y1,pressed,points, subsetImage

    subsetImage = img

    pressed = False # True if mouse is pressed
    point_x1 , point_y1 = None , None

    points =[]

    cv.namedWindow(name)
    cv.setMouseCallback(name,drawLineMouse)

    while(1):
        cv.imshow(name,subsetImage)
        if cv.waitKey(1) & 0xFF == 13:
            break
    cv.destroyAllWindows()

    return points

########################################################################################################################
# Function - try integer value
########################################################################################################################
def tryint(s):
    try:
        return int(s)
    except:
        return s

########################################################################################################################
# Turn a string into a list of string
########################################################################################################################
def stringToList(s):

    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

########################################################################################################################
# Function to select the captured images folder
########################################################################################################################
def captured_folder(capturedFolder, captured_status, console, canvas):
    global filename_captured, test_captured, fileNames, Format, Images

    filename_captured = filedialog.askdirectory()
    capturedFolder.set(filename_captured)

    console.insert(tk.END,'###########################################################################################\n\n')
    console.see('insert')

    if not filename_captured:
        captured_status.configure(bg = 'red') # Red indicator
        console.insert(tk.END, 'The captured folder was not selected\n\n')
        console.see('insert')
        messagebox.showerror('Error','The captured folder was not selected!')
    else:
        captured_status.configure(bg = '#00cd00') # Green indicator
        console.insert(tk.END, f'Image captured folder - {capturedFolder.get()}\n\n')
        console.see('insert')

        # Captured image files:
        fileNames = sorted(glob.glob(filename_captured+'\\*'),key=stringToList)
        Format = '.'+fileNames[0].rsplit('.', 1)[1]
        Images = len(fileNames)

        fig = plt.figure()
        ax = fig.gca()
        dxplot = int(cv.imread(fileNames[0]).shape[1])
        dyplot = int(cv.imread(fileNames[0]).shape[0])

        ratio_plot = dxplot/dyplot

        if ratio_plot <= 1.33333333:

            dxplotdark = dyplot*1.33333333
            dyplotdark = dyplot
            ax.imshow(cv.imread(fileNames[0]), zorder=2)
            ax.plot([0-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2, dxplotdark-dxplotdark/2+dxplot/2, dxplotdark-
                             dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2],[0, dyplotdark, dyplotdark, 0, 0],'black',
                             zorder=1)
            ax.fill_between([0-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2, dxplotdark-dxplotdark/2+dxplot/2,
                             dxplotdark-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2],[0, dyplotdark,
                             dyplotdark, 0, 0], color='black', zorder=1)
            ax.axis('off')
            plt.xlim(-dxplotdark/2+dxplot/2,dxplotdark-dxplotdark/2+dxplot/2)
            plt.ylim(0,dyplotdark)
            plt.subplots_adjust(0,0,1,1,0,0)

        else:
            dxplotdark = dxplot
            dyplotdark = dxplot/1.33333333
            ax.imshow(cv.imread(fileNames[0]), zorder=2)
            ax.plot([0, 0, dxplotdark, dxplotdark, 0],[0-dyplotdark/2+dyplot/2, +dyplotdark-dyplotdark/2+dyplot/2,
                    +dyplotdark-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2],'black',
                    zorder=1)
            ax.fill_between([0, 0, dxplotdark, dxplotdark, 0],[0-dyplotdark/2+dyplot/2, +dyplotdark-dyplotdark/2+
                    dyplot/2, +dyplotdark-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2],
                    color='black', zorder=1)
            ax.axis('off')
            plt.xlim(0,dxplotdark)
            plt.ylim(-dyplotdark/2+dyplot/2,dyplotdark-dyplotdark/2+dyplot/2)
            plt.subplots_adjust(0,0,1,1,0,0)

        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        image = np.flip(image.reshape((h, w, 3)), axis=0)

        plt.cla()
        plt.clf()

        image = cv.resize(image, (640, 480))
        cv.putText(image,'REFERENCE IMAGE',(20,30),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv.LINE_AA)
        image = ImageTk.PhotoImage (Image.fromarray (image))

        canvas.image = image
        canvas.configure(image = image)

        console.insert(tk.END, f'{Images} images were imported with {Format} format\n\n')
        console.see('insert')
        test_captured = True

########################################################################################################################
# Save function - DIC parameters
########################################################################################################################
def save(menu, console, file_var, V, file, capturedFolder, SubIr, SubIb, Nx, Ny, Valpixel, Valmm, Opi, OpiSub,
         Version, TypeCut, NumCut, Adjust, Alpha, Method, Correlation, Criterion, Step, Interpolation, Filtering,
         Kernel):

    if file_var.get():

        f = open(file.get(),"w")

        f.write('iCorrVision-2D Correlation Module - '+str(datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S")))
        f.write('\nImage captured folder:\n')
        f.write(str(capturedFolder.get().rstrip("\n")))
        f.write('\nReference subset size:\n')
        f.write(str(SubIr.get()))
        f.write('\nSearch subset size:\n')
        f.write(str(SubIb.get()))
        f.write('\nSubset step:\n')
        f.write(str(Step.get()))
        f.write('\nNumber of steps in x direction:\n')
        f.write(str(Nx.get()))
        f.write('\nNumber of steps in y direction:\n')
        f.write(str(Ny.get()))
        f.write('\nValue in pixels for calibration:\n')
        f.write(str(Valpixel.get()))
        f.write('\nValue in mm for calibration:\n')
        f.write(str(Valmm.get()))
        f.write('\nInterpolation type:\n')
        f.write(str(Interpolation.get().rstrip("\n")))
        f.write('\nImage interpolation factor (kb):\n')
        f.write(str(Opi.get()))
        f.write('\nSubpixel level (ka):\n')
        f.write(str(OpiSub.get()))
        f.write('\nCorrelation type:\n')
        f.write(str(Correlation.get().rstrip("\n")))
        f.write('\nFiltering:\n')
        f.write(str(Filtering.get().rstrip("\n")))
        f.write('\nKernel size:\n')
        f.write(str(Kernel.get()))
        f.write('\nMethod:\n')
        f.write(str(Method.get().rstrip("\n")))
        f.write('\nCorrelation criterion:\n')
        f.write(str(Criterion.get()))
        f.write('\nSelected version:\n')
        f.write(str(Version.get().rstrip("\n")))
        f.write('\nType of mesh cut:\n')
        f.write(str(TypeCut.get().rstrip("\n")))
        f.write('\nNumber of cuts:\n')
        f.write(str(NumCut.get()))
        f.write('\nContrast adjust:\n')
        f.write(str(Adjust.get()))
        f.write('\nAngle of rotation:\n')
        f.write(str(Alpha.get()))
        f.close()

        console.insert(tk.END,
                       '###########################################################################################\n\n')
        console.insert(tk.END, f'Data was successfully saved in {file.get()}\n\n')
        console.see('insert')

    else:

        save_as(menu, console, file_var, V, file, capturedFolder, SubIr, SubIb, Nx, Ny, Valpixel, Valmm, Opi, OpiSub,
                Version, TypeCut, NumCut, Adjust, Alpha, Method, Correlation, Criterion, Step, Interpolation, Filtering,
                Kernel)

########################################################################################################################
# Save as function - DIC parameters
########################################################################################################################
def save_as(menu, console, file_var, V, file, capturedFolder, SubIr, SubIb, Nx, Ny, Valpixel, Valmm, Opi, OpiSub,
            Version, TypeCut, NumCut, Adjust, Alpha, Method, Correlation, Criterion, Step, Interpolation, Filtering,
            Kernel):

    file_var.set(True)

    console.insert(tk.END, 'Indicate a .dat file to save the DIC parameters\n\n')
    console.see('insert')

    file.set(filedialog.asksaveasfilename())

    menu.title('iCorrVision-2D Correlation Module '+V+' - '+ntpath.basename(file.get()))

    f = open(file.get(),"w+")

    f.write('iCorrVision-2D Correlation Module - '+str(datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S")))
    f.write('\nImage captured folder:\n')
    f.write(str(capturedFolder.get().rstrip("\n")))
    f.write('\nReference subset size:\n')
    f.write(str(SubIr.get()))
    f.write('\nSearch subset size:\n')
    f.write(str(SubIb.get()))
    f.write('\nSubset step:\n')
    f.write(str(Step.get()))
    f.write('\nNumber of steps in x direction:\n')
    f.write(str(Nx.get()))
    f.write('\nNumber of steps in y direction:\n')
    f.write(str(Ny.get()))
    f.write('\nValue in pixels for calibration:\n')
    f.write(str(Valpixel.get()))
    f.write('\nValue in mm for calibration:\n')
    f.write(str(Valmm.get()))
    f.write('\nInterpolation type:\n')
    f.write(str(Interpolation.get().rstrip("\n")))
    f.write('\nImage interpolation factor (kb):\n')
    f.write(str(Opi.get()))
    f.write('\nSubpixel level (ka):\n')
    f.write(str(OpiSub.get()))
    f.write('\nCorrelation type:\n')
    f.write(str(Correlation.get().rstrip("\n")))
    f.write('\nFiltering:\n')
    f.write(str(Filtering.get().rstrip("\n")))
    f.write('\nKernel size:\n')
    f.write(str(Kernel.get()))
    f.write('\nMethod:\n')
    f.write(str(Method.get().rstrip("\n")))
    f.write('\nCorrelation criterion:\n')
    f.write(str(Criterion.get()))
    f.write('\nSelected version:\n')
    f.write(str(Version.get().rstrip("\n")))
    f.write('\nType of mesh cut:\n')
    f.write(str(TypeCut.get().rstrip("\n")))
    f.write('\nNumber of cuts:\n')
    f.write(str(NumCut.get()))
    f.write('\nContrast adjust:\n')
    f.write(str(Adjust.get()))
    f.write('\nAngle of rotation:\n')
    f.write(str(Alpha.get()))
    f.close()

    console.insert(tk.END,
                   '###########################################################################################\n\n')
    console.insert(tk.END, f'Data was successfully saved in {file.get()}\n\n')
    console.see('insert')

########################################################################################################################
# Load function - DIC parameters
########################################################################################################################
def load(menu, captured_status, console, canvas, file_var, V, file, capturedFolder, SubIr, SubIb, Nx, Ny, Valpixel,
         Valmm, Opi, OpiSub, Version, TypeCut, NumCut, Adjust, Alpha, Method, Correlation, Criterion, Step,
         Interpolation, Filtering, Kernel):

    global test_captured, fileNames, Format, Images

    file_load = filedialog.askopenfilename()

    if file_load != '':

        menu.title('iCorrVision-2D Correlation Module '+V+' - '+ntpath.basename(file_load))

        file.set(file_load)

        l = open(file_load,"r")
        w = 2
        lines = l.readlines()
        capturedFolder.set(lines[w].rstrip("\n")); w = w + 2
        SubIr.set(lines[w]); w = w + 2
        SubIb.set(lines[w]); w = w + 2
        Step.set(lines[w]); w = w + 2
        Nx.set(lines[w]); w = w + 2
        Ny.set(lines[w]); w = w + 2
        Valpixel.set(lines[w]); w = w + 2
        Valmm.set(lines[w]); w = w + 2
        Interpolation.set(lines[w].rstrip("\n")); w = w + 2
        Opi.set(lines[w]); w = w + 2
        OpiSub.set(lines[w]); w = w + 2
        Correlation.set(lines[w].rstrip("\n")); w = w + 2
        Filtering.set(lines[w].rstrip("\n")); w = w + 2
        Kernel.set(lines[w]); w = w + 2
        Method.set(lines[w].rstrip("\n")); w = w + 2
        Criterion.set(lines[w]); w = w + 2
        Version.set(lines[w].rstrip("\n")); w = w + 2
        TypeCut.set(lines[w].rstrip("\n")); w = w + 2
        NumCut.set(lines[w]); w = w + 2
        Adjust.set(lines[w]); w = w + 2
        Alpha.set(lines[w])

        captured_status.configure(bg = '#00cd00') # Green indicator
        console.insert(tk.END,
                       '###########################################################################################\n\n')
        console.insert(tk.END, f'Image captured folder - {capturedFolder.get()}\n\n')
        console.see('insert')

        fileNames = sorted(glob.glob(capturedFolder.get()+'\\*'),key=stringToList)
        Format = '.'+fileNames[0].rsplit('.', 1)[1]
        Images = len(fileNames)

        fig = plt.figure()
        ax = fig.gca()
        dxplot = int(cv.imread(fileNames[0]).shape[1])
        dyplot = int(cv.imread(fileNames[0]).shape[0])

        ratio_plot = dxplot/dyplot;

        if ratio_plot <= 1.33333333:

            dxplotdark = dyplot*1.33333333
            dyplotdark = dyplot
            ax.imshow(cv.imread(fileNames[0]), zorder=2)
            ax.plot([0-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2, dxplotdark-dxplotdark/2+dxplot/2, dxplotdark-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2],[0, dyplotdark, dyplotdark, 0, 0],'black', zorder=1)
            ax.fill_between([0-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2, dxplotdark-dxplotdark/2+dxplot/2, dxplotdark-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2],[0, dyplotdark, dyplotdark, 0, 0], color='black', zorder=1)
            ax.axis('off')
            plt.xlim(-dxplotdark/2+dxplot/2,dxplotdark-dxplotdark/2+dxplot/2)
            plt.ylim(0,dyplotdark)
            plt.subplots_adjust(0,0,1,1,0,0)

        else:
            dxplotdark = dxplot
            dyplotdark = dxplot/1.33333333
            ax.imshow(cv.imread(fileNames[0]), zorder=2)
            ax.plot([0, 0, dxplotdark, dxplotdark, 0],[0-dyplotdark/2+dyplot/2, +dyplotdark-dyplotdark/2+dyplot/2, +dyplotdark-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2],'black', zorder=1)
            ax.fill_between([0, 0, dxplotdark, dxplotdark, 0],[0-dyplotdark/2+dyplot/2, +dyplotdark-dyplotdark/2+dyplot/2, +dyplotdark-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2], color='black', zorder=1)
            ax.axis('off')
            plt.xlim(0,dxplotdark)
            plt.ylim(-dyplotdark/2+dyplot/2,dyplotdark-dyplotdark/2+dyplot/2)
            plt.subplots_adjust(0,0,1,1,0,0)

        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        image = np.flip(image.reshape((h, w, 3)),axis=0)

        plt.cla()
        plt.clf()
        ax = None
        fig = None

        image = cv.resize(image, (640, 480))
        cv.putText(image,'REFERENCE IMAGE',(20,30),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv.LINE_AA)
        image = ImageTk.PhotoImage (Image.fromarray (image))

        canvas.image = image
        canvas.configure(image = image)

        console.insert(tk.END, f'{Images} images were imported with {Format} format\n\n')
        console.see('insert')

        test_captured = True
        file_var.set(True)

    else:

        console.insert(tk.END, f'No files were selected \n\n')
        console.see('insert')

########################################################################################################################
# Function to clear data and restore GUI
########################################################################################################################
def clear(menu, CurrentDir, captured_status, console, console_process, progression, progression_bar, canvas,
          canvas_text, file_var, V, file, capturedFolder, SubIr, SubIb, Nx, Ny, Valpixel, Valmm, Opi, OpiSub, Version,
          TypeCut, NumCut, Adjust, Alpha, Method, Correlation, Criterion, Step, Interpolation, Filtering, Kernel):

    file_var.set(False)

    menu.title('iCorrVision-2D Correlation Module '+V)

    console.delete('1.0', END)
    console_process.delete('1.0', END)

    capturedFolder.set(''); captured_status.configure(bg = 'red') # Red indicator
    SubIr.set(0)
    SubIb.set(0)
    Step.set(0)
    Nx.set(0)
    Ny.set(0)
    Valpixel.set(0)
    Valmm.set(0.0)
    Opi.set(0)
    OpiSub.set('None')
    Adjust.set(0.0)
    Criterion.set(0.0)
    NumCut.set(0)
    Alpha.set(0.0)
    Version.set('Select')
    TypeCut.set('None')
    Correlation.set('Select')
    Method.set('Select')
    Interpolation.set('Select')
    Filtering.set('Select')
    Kernel.set(0)

    image_black = Image.open(CurrentDir+'\static\ImageBlack.tiff')
    image_black = image_black.resize((640, 480), Image.ANTIALIAS)
    image_black_re = ImageTk.PhotoImage(image_black)
    canvas.image_black_re = image_black_re
    canvas.configure(image = image_black_re)

    console.insert(tk.END,
                   f'#################################################################################  {V}\n\n'
                   '                          **  iCorrVision-2D Correlation Module **                         \n\n'
                   '###########################################################################################\n\n')

    console.insert(tk.END,
                   'Please load project or select the image captured folder and DIC settings\n\n')
    console.see('insert')

    progression.coords(progression_bar, 0, 0, 0, 25); progression.itemconfig(canvas_text, text='')

########################################################################################################################
# Circular ROI function
########################################################################################################################
def CircularROI_DIC(menu, console, title, img):
    # Module global variables:
    global fileNames, points, pressed, text, xCenter, yCenter, dxCirc, dyCirc, ptn

    # Set drawing mode to False:
    pressed = False

    text = title

    # Mouse events:
    def click_event_Circular_DIC(event, x, y, flags, param):
        global points, pressed, text, ptn

        # Save previous state on cache:
        cache = img.copy()

        if event == cv.EVENT_LBUTTONDOWN:
            # Clear points:
            ptn = []
            points = []

            # Set drawing mode to True:
            pressed = True

            # First vertices construction:
            points.append((x, y))  # Save the location of first vertices

        elif event == cv.EVENT_MOUSEMOVE:
            if pressed == True:

                xPL = points[-1][0]
                yPL = points[-1][1]

                xPR = x
                yPR = y

                xCenter = int((xPR + xPL) / 2)
                yCenter = int((yPR + yPL) / 2)

                dxCirc = abs(int((xPR - xPL) / 2))
                dyCirc = abs(int((yPR - yPL) / 2))

                cv.ellipse(cache, (xCenter, yCenter), (dxCirc, dyCirc), 0, 0, 360, 255, 2)

                cv.imshow(text, cache)

        elif event == cv.EVENT_LBUTTONUP:

            # Set drawing mode to False:
            pressed = False

            xPL = points[-1][0]
            yPL = points[-1][1]

            xPR = x
            yPR = y

            xCenter = int((xPR + xPL) / 2)
            yCenter = int((yPR + yPL) / 2)

            dxCirc = abs(int((xPR - xPL) / 2))
            dyCirc = abs(int((yPR - yPL) / 2))

            cv.ellipse(cache, (xCenter, yCenter), (dxCirc, dyCirc), 0, 0, 360, (255, 0, 0), -1)

            alpha = 0.4  # Transparency factor.

            # Following line overlays transparent rectangle over the image
            cache = cv.addWeighted(cache, alpha, img, 1 - alpha, 0)

            t = np.linspace(0, 2 * pi, 100)
            xEl = xCenter + dxCirc * np.cos(t)
            yEl = yCenter + dyCirc * np.sin(t)

            for w in range(0,100):
                ptn.append((int(xEl[w]),int(yEl[w])))
                #cv.circle(cache, [int(xEl[w]),int(yEl[w])], 2, 255, -1)

            cv.imshow(text, cache)

    # Routine created to check the captured images:
    try:
        fileNames
    except NameError:
        console.insert(tk.END, 'No images were found. Make sure the images have been imported!\n\n')
        console.see('insert')
        messagebox.showerror('Error', 'No images were found. Make sure the images have been imported!')
    else:

        cv.imshow(title, img)
        points = []
        ptn = []

        cv.setMouseCallback(title, click_event_Circular_DIC)

        cv.waitKey(0)  # The USER have to press any key to exit the interface (ENTER)
        cv.destroyAllWindows()

    return ptn

########################################################################################################################
# ROI function
########################################################################################################################
def SelectROI_DIC(menu, console, title, img):
    # Module global variables:
    global fileNames, points, pressed, text

    # Set drawing mode to False:
    pressed = False

    text = title

    # Mouse events:
    def click_event_DIC(event, x, y, flags, param):
        global points, pressed, text

        # Save previous state on cache:
        cache = img_measure.copy()

        if event == cv.EVENT_LBUTTONDOWN:
            # Clear points:
            points = []

            # Set drawing mode to True:
            pressed = True

            # First vertice construction:
            points.append((x, y))  # Save the location of first vertice

        elif event == cv.EVENT_MOUSEMOVE:
            if pressed == True:
                # First and second vertices construction:
                cv.rectangle(cache, points[-1],(x, y), (0, 0, 255), 1, cv.LINE_AA)
                cv.line(cache, (points[-1][0],int((y+points[-1][1])/2)), (x,int((y+points[-1][1])/2)),(255, 0, 0), 1,
                        cv.LINE_AA)
                cv.line(cache, (int((x + points[-1][0]) / 2), points[-1][1]), (int((x + points[-1][0]) / 2), y),
                        (255, 0, 0), 1, cv.LINE_AA)
                cv.rectangle(cache, (0,0), (int(cache.shape[1]),30), (255, 255, 255), -1)
                cv.putText(cache, f'(x = {x}, y = {y}) - ROI SIZE = {x-points[-1][0]} x {y-points[-1][1]}', (10,20),
                           cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv.LINE_AA)
                # Display the ruler on cache image:
                cv.imshow(text, cache)

        elif event == cv.EVENT_LBUTTONUP:

            # Set drawing mode to False:
            pressed = False

            # Display the line used to measure Valpixel:
            cv.rectangle(cache, points[-1], (x, y), (0, 0, 255), 2, cv.FILLED)
            cv.rectangle(cache, (0, 0), (int(cache.shape[1]), 30), (255, 255, 255), -1)
            cv.putText(cache, f'(x = {x}, y = {y}) - ROI SIZE = {x - points[-1][0]} x {y - points[-1][1]}', (10, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
            points.append((x, y))
            cv.imshow(text, cache)

    # Routine created to check the captured images:
    try:
        fileNames
    except NameError:
        console.insert(tk.END, 'No images were found. Make sure the images have been imported!\n\n')
        console.see('insert')
        messagebox.showerror('Error', 'No images were found. Make sure the images have been imported!')
    else:

        dxfigure = int(img.shape[1])
        dyfigure = int(img.shape[0])

        screen_height = menu.winfo_screenheight() - 100

        # Change size of displayed image:
        if dyfigure > screen_height:
            ratio = screen_height / dyfigure
            img_measure = cv.resize(img, (int(screen_height * dxfigure / dyfigure), screen_height))
            cv.line(img_measure, (0, int(dyfigure / 2)), (dxfigure, int(dyfigure / 2)), (255, 255, 255), 1,
                    cv.LINE_AA)
            cv.line(img_measure, (int(dxfigure / 2), 0), (int(dxfigure / 2), dyfigure), (255, 255, 255), 1,
                    cv.LINE_AA)
            cv.imshow(title, img_measure)
            points = []
            cv.setMouseCallback(title, click_event_DIC)

        else:
            img_measure = img
            cv.line(img_measure, (0, int(dyfigure / 2)), (dxfigure, int(dyfigure / 2)), (255, 255, 255), 1,
                    cv.LINE_AA)
            cv.line(img_measure, (int(dxfigure / 2), 0), (int(dxfigure / 2), dyfigure), (255, 255, 255), 1,
                    cv.LINE_AA)
            cv.imshow(title, img_measure)
            points = []
            cv.setMouseCallback(title, click_event_DIC)

        cv.waitKey(0)  # The USER have to press any key to exit the interface (ENTER)
        cv.destroyAllWindows()

        ptn = np.zeros((2, 2))

        # Points location correction due to image correction:
        if dyfigure > screen_height:
            for i in [0, 1]:
                for j in [0, 1]:
                    ptn[i][j] = points[i][j] / ratio

        else:
            ptn = points

    xinitline = ptn[0][0]
    yinitline = ptn[0][1]
    dxline = ptn[1][0]-ptn[0][0]
    dyline = ptn[1][1]-ptn[0][1]

    return xinitline,yinitline,dxline,dyline

########################################################################################################################
# Function to measure distance in pixels
########################################################################################################################
def measure(menu, console, Valpixel, Valmm, Adjust, Alpha):
    # Module global variables:
    global fileNames, points, drawing

    # Set drawing mode to False:
    drawing = False

    # Mouse events:
    def click_event(event, x, y, flags, param):
        global points, drawing

        # Save previous state on cache:
        cache = img_measure.copy()

        if event == cv.EVENT_LBUTTONDOWN:
            # Clear points:
            points = []

            # Set drawing mode to True:
            drawing = True

            # First vertice construction:
            cv.circle(cache, (x, y), 2, (0, 0, 255), 2, cv.FILLED)
            cv.imshow('Measure Valpixel - press enter key', cache)
            points.append((x, y)) # Save the location of first vertice

        elif event == cv.EVENT_MOUSEMOVE:
            if drawing == True:

                # First ans second vertices construction:
                cv.circle(cache, points[-1], 2, (0, 0, 255), 2, cv.FILLED)
                cv.circle(cache, (x, y), 2, (0, 0, 255), 2, cv.FILLED)
                cv.line(cache, points[-1], (x, y), (255, 0, 0), 1, cv.LINE_AA)

                # Lenght of constructed line:
                D = np.sqrt((points[0][0]-x)**2 + (points[0][1]-y)**2)

                # Lenght of perpendicular lines
                Len = D/4

                # Construction of perpendicular lines vertices:
                cv.circle(cache, (
                int(points[0][0] + Len * (points[0][1] - y) / D), int(points[0][1] + Len * (x - points[0][0]) / D)), 2,
                          (0, 0, 255), 2, cv.FILLED)
                cv.circle(cache, (
                int(points[0][0] - Len * (points[0][1] - y) / D), int(points[0][1] - Len * (x - points[0][0]) / D)), 2,
                          (0, 0, 255), 2, cv.FILLED)

                cv.circle(cache, (
                int(x + Len * (points[0][1] - y) / D), int(y + Len * (x - points[0][0]) / D)), 2,
                          (0, 0, 255), 2, cv.FILLED)
                cv.circle(cache, (
                int(x - Len * (points[0][1] - y) / D), int(y - Len * (x - points[0][0]) / D)), 2,
                          (0, 0, 255), 2, cv.FILLED)

                # Construction of perpendicular lines:
                cv.line(cache, (
                int(points[0][0] + Len * (points[0][1] - y) / D), int(points[0][1] + Len * (x - points[0][0]) / D)),
                        (int(points[0][0] - Len * (points[0][1] - y) / D),
                         int(points[0][1] - Len * (x - points[0][0]) / D)),
                        (0, 0, 255), 1, cv.LINE_AA)

                cv.line(cache, (
                int(x + Len * (points[0][1] - y) / D), int(y + Len * (x - points[0][0]) / D)),
                        (int(x - Len * (points[0][1] - y) / D),
                         int(y - Len * (x - points[0][0]) / D)),
                        (0, 0, 255), 1, cv.LINE_AA)

                # Display the ruler on cache image:
                cv.imshow('Measure Valpixel - press enter key', cache)

        elif event == cv.EVENT_LBUTTONUP:

            # Set drawing mode to False:
            drawing = False

            # Display the line used to measure Valpixel:
            cv.circle(cache, points[-1], 2, (0, 0, 255), 2, cv.FILLED)
            cv.circle(cache, (x, y), 2, (0, 0, 255), 2, cv.FILLED)
            points.append((x, y))
            cv.line(cache, points[-1], points[-2], (255, 0, 0), 1, cv.LINE_AA)
            cv.imshow('Measure Valpixel - press enter key', cache)

    # Routine created to check the captured images:
    try:
        fileNames
    except NameError:
        console.insert(tk.END, 'Import the captured images before measuring the distance in pixels!\n\n')
        console.see('insert')
        messagebox.showerror('Error','Import the captured images before measuring the distance in pixels!')
    else:

        if Adjust.get() == 0.0:
            adjust = cv.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
        else:
            adjust = cv.createCLAHE(clipLimit=Adjust.get(), tileGridSize=(8,8))

        dxfigure = int(cv.imread(fileNames[0]).shape[1])
        dyfigure = int(cv.imread(fileNames[0]).shape[0])

        screen_height = menu.winfo_screenheight()-100

        # Change size of displayed image:
        if dyfigure > screen_height:
            ratio = screen_height/dyfigure
            img_measure = adjust.apply(cv.resize(cv.imread(fileNames[0],0), (int(screen_height*dxfigure/dyfigure), screen_height)))
            img_measure = cv.cvtColor(img_measure,cv.COLOR_GRAY2RGB)
            img_measure = ndimage.rotate(img_measure, Alpha.get(), reshape=False)
            cv.imshow('Measure Valpixel - press enter key', img_measure)
            points = []
            cv.setMouseCallback('Measure Valpixel - press enter key', click_event)

        else:
            img_measure = adjust.apply(cv.imread(fileNames[0],0))
            img_measure = cv.cvtColor(img_measure,cv.COLOR_GRAY2RGB)
            img_measure = ndimage.rotate(img_measure, Alpha.get(), reshape=False)
            cv.imshow('Measure Valpixel - press enter key', img_measure)
            points = []
            cv.setMouseCallback('Measure Valpixel - press enter key', click_event)

        cv.waitKey(0) # The USER have to press any key to exit the interface (ENTER)
        cv.destroyAllWindows()

        ptn = np.zeros((2,2))

        # Points location correction due to image correction:
        if dyfigure > screen_height:
            for i in [0,1]:
                for j in [0, 1]:
                     ptn[i][j] = points[i][j]/ratio

        else:
            ptn[i][j] = points[i][j]

        # Find Valpixel value from interface:
        Valpixel.set(np.sqrt((ptn[0][0]-ptn[1][0])**2 + (ptn[0][1]-ptn[1][1])**2))

########################################################################################################################
# Function to check the contrast according to the contrast factor
########################################################################################################################
def verify(Adjust, canvas, console, Alpha):
    global fileNames, test_constrast

    console.insert(tk.END,
                   '###########################################################################################\n\n')
    console.see('insert')

    try:
        fileNames
    except NameError:
        console.insert(tk.END, 'Import the captured images before running the contrast adjustment test!\n\n')
        console.see('insert')
        messagebox.showerror('Error','Import the captured images before running the contrast adjustment test!')
    else:

        adjust = cv.createCLAHE(clipLimit=Adjust.get(), tileGridSize=(8,8))

        fig = plt.figure()
        ax = fig.gca()
        dxplot = int(cv.imread(fileNames[0]).shape[1])
        dyplot = int(cv.imread(fileNames[0]).shape[0])

        ratio_plot = dxplot/dyplot;

        if ratio_plot <= 1.33333333:

            dxplotdark = dyplot*1.33333333
            dyplotdark = dyplot
            ax.imshow(cv.imread(fileNames[0]), zorder=2)
            ax.plot([0-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2, dxplotdark-dxplotdark/2+dxplot/2, dxplotdark
                    -dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2],[0, dyplotdark, dyplotdark, 0, 0],'black',
                    zorder=1)
            ax.fill_between([0-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2, dxplotdark-dxplotdark/2+dxplot/2,
                    dxplotdark-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2],[0, dyplotdark, dyplotdark, 0, 0],
                    color='black', zorder=1)
            ax.axis('off')
            plt.xlim(-dxplotdark/2+dxplot/2,dxplotdark-dxplotdark/2+dxplot/2)
            plt.ylim(0,dyplotdark)
            plt.subplots_adjust(0,0,1,1,0,0)

        else:
            dxplotdark = dxplot
            dyplotdark = dxplot/1.33333333
            ax.imshow(cv.imread(fileNames[0]), zorder=2)
            ax.plot([0, 0, dxplotdark, dxplotdark, 0],[0-dyplotdark/2+dyplot/2, +dyplotdark-dyplotdark/2+dyplot/2,
                     +dyplotdark-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2],'black',
                     zorder=1)
            ax.fill_between([0, 0, dxplotdark, dxplotdark, 0],[0-dyplotdark/2+dyplot/2, +dyplotdark-dyplotdark/2+
                     dyplot/2, +dyplotdark-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2],
                     color='black', zorder=1)
            ax.axis('off')
            plt.xlim(0,dxplotdark)
            plt.ylim(-dyplotdark/2+dyplot/2,dyplotdark-dyplotdark/2+dyplot/2)
            plt.subplots_adjust(0,0,1,1,0,0)

        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        image = np.flip(image.reshape((h, w, 3)),axis=0)

        plt.cla()
        plt.clf()

        image = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
        image = ndimage.rotate(image, Alpha.get(), reshape=False)
        image = adjust.apply(cv.resize(image, (640, 480)))
        image = cv.cvtColor(image,cv.COLOR_GRAY2RGB)
        cv.putText(image,'REFERENCE IMAGE',(20,30),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv.LINE_AA)
        image = ImageTk.PhotoImage (Image.fromarray (image))

        canvas.image = image
        canvas.configure(image = image)

        console.insert(tk.END, f'All images will be adjusted according to the given contrast factor of {Adjust.get()}\n\n')
        console.see('insert')

        test_constrast = True

########################################################################################################################
# Function to close the software
########################################################################################################################
def close(menu):
    global t2
    ans = messagebox.askquestion('Close','Are you sure you want to exit iCorrVision-2D Correlation module?',icon ='question')
    if ans == 'yes':

        menu.destroy()
        menu.quit()

########################################################################################################################
# Abort function
########################################################################################################################
def abort(abort_param):

    abort_param.set(True)

########################################################################################################################
# 2D Digital Image Correlation function V1
########################################################################################################################
def Corr2D_V1(Image0, Image1, SubIr, SubIb, xL_mem, xL_shape, yL_mem, yL_shape, uL_mem, uL_shape, vL_mem,
    vL_shape, Iun, Id, Type, Method, Criterion, i0, i1, j0, j1):
    global process_x0_mem, process_x0_shape, process_y0_mem, process_y0_shape
    global process_u0_mem, process_u0_shape, process_v0_mem, process_v0_shape

    process_x0_mem = xL_mem
    process_x0_shape = xL_shape
    process_y0_mem = yL_mem
    process_y0_shape = yL_shape
    process_u0_mem = uL_mem
    process_u0_shape = uL_shape
    process_v0_mem = vL_mem
    process_v0_shape = vL_shape

    x0 = np.frombuffer(process_x0_mem, dtype=np.float64).reshape(process_x0_shape)
    y0 = np.frombuffer(process_y0_mem, dtype=np.float64).reshape(process_y0_shape)
    u0 = np.frombuffer(process_u0_mem, dtype=np.float64).reshape(process_u0_shape)
    v0 = np.frombuffer(process_v0_mem, dtype=np.float64).reshape(process_v0_shape)

    for i in range(i0, i1):
        for j in range(j0, j1):
            # Test nan values:
            if np.isnan(x0[Image0][i][j]):
                # If it exists, node ij assumes value nan:
                x0[Image1][i][j] = float('nan');
                y0[Image1][i][j] = float('nan');
                u0[Image1][i][j] = float('nan');
                v0[Image1][i][j] = float('nan')
            else:
                # Estimate the first node position for the reference and search images:
                xrmin = int(x0[Image0][i][j] - (SubIr) / 2)
                yrmin = int(y0[Image0][i][j] - (SubIr) / 2)
                xbmin = int(x0[Image0][i][j] - (SubIb) / 2)
                if xbmin < 0:
                    xbmin = 0
                    SubIb_x = x0[Image0][i][j] * 2
                    # Position of the correlation matrix centroid:
                    xpeak_ac = (SubIb_x - SubIr) / 2
                    testexp = True
                else:
                    # Position of the correlation matrix centroid:
                    xpeak_ac = (SubIb - SubIr) / 2
                    testexp = False

                ybmin = int(y0[Image0][i][j] - (SubIb) / 2)
                if ybmin < 0:
                    ybmin = 0
                    SubIb_y = y0[Image0][i][j] * 2

                    ypeak_ac = (SubIb_y - SubIr) / 2
                    testeyp = True
                else:
                    ypeak_ac = (SubIb - SubIr) / 2
                    testeyp = False

                # Crop reference and search images according to SubIr and SubIb values:
                Ir = Iun[yrmin:yrmin + SubIr, xrmin:xrmin + SubIr]
                Ib = Id[ybmin:ybmin + SubIb, xbmin:xbmin + SubIb]

                try:
                    # Correlation:
                    c = cv.matchTemplate(Ib, Ir, Method)

                    # Search for the maximum coefficient of correlation and its position:
                    _, maxVal, _, maxc = cv.minMaxLoc(c)

                    # If the maximum coefficient of correlation is lower than the correlation criterion, the node ij assumes the
                    # value nan, indicating the loss of correlation:
                    if maxVal < Criterion:

                        ui = float('nan')
                        vi = float('nan')

                    else:

                        xc = np.linspace(maxc[0] - 2, maxc[0] + 2, 5)
                        yc = np.linspace(maxc[1] - 2, maxc[1] + 2, 5)

                        cc = c[maxc[1] - 2:maxc[1] + 3, maxc[0] - 2:maxc[0] + 3]

                        xcint = xc;
                        ycint = yc;

                        f = interp2d(xc, yc, cc, kind='cubic')

                        del cc

                        ccinterp = f(xcint, ycint)

                        maxc_interp = np.unravel_index(ccinterp.argmax(), ccinterp.shape)

                        xpeak = xcint[maxc_interp[0]];
                        ypeak = ycint[maxc_interp[1]]

                        # Displacement calculation:
                        ui = xpeak - xpeak_ac
                        vi = ypeak - ypeak_ac

                except:
                    ui = float('nan')
                    vi = float('nan')

                if Type == 'Lagrangian':

                    x0[Image1][i][j] = x0[Image0][i][j] + ui
                    y0[Image1][i][j] = y0[Image0][i][j] + vi
                    u0[Image1][i][j] = ui;
                    v0[Image1][i][j] = vi;

                else:

                    x0[Image1][i][j] = x0[Image0][i][j]
                    y0[Image1][i][j] = y0[Image0][i][j]
                    u0[Image1][i][j] = ui;
                    v0[Image1][i][j] = vi;

########################################################################################################################
# 2D Digital Image Correlation function V2
########################################################################################################################
def Corr2D_V2(Image0, Image1, SubIr, SubIb, OpiSub, xL_mem, xL_shape, yL_mem, yL_shape, uL_mem, uL_shape,
    vL_mem, vL_shape, Iun, Id, Type, Method, Criterion,i0,i1,j0,j1):
    global process_x0_mem, process_x0_shape, process_y0_mem, process_y0_shape
    global process_u0_mem, process_u0_shape, process_v0_mem, process_v0_shape

    process_x0_mem = xL_mem
    process_x0_shape = xL_shape
    process_y0_mem = yL_mem
    process_y0_shape = yL_shape
    process_u0_mem = uL_mem
    process_u0_shape = uL_shape
    process_v0_mem = vL_mem
    process_v0_shape = vL_shape

    x0 = np.frombuffer(process_x0_mem, dtype=np.float64).reshape(process_x0_shape)
    y0 = np.frombuffer(process_y0_mem, dtype=np.float64).reshape(process_y0_shape)
    u0 = np.frombuffer(process_u0_mem, dtype=np.float64).reshape(process_u0_shape)
    v0 = np.frombuffer(process_v0_mem, dtype=np.float64).reshape(process_v0_shape)

    for i in range(i0, i1):
        for j in range(j0, j1):
            # Test nan values:
            if np.isnan(x0[Image0][i][j]):
                # If it exists, node ij assumes value nan:
                x0[Image1][i][j] = float('nan');
                y0[Image1][i][j] = float('nan');
                u0[Image1][i][j] = float('nan');
                v0[Image1][i][j] = float('nan')
            else:
                # Estimate the first node position for the reference and search images:
                xrmin = int(x0[Image0][i][j] - (SubIr) / 2)
                yrmin = int(y0[Image0][i][j] - (SubIr) / 2)
                xbmin = int(x0[Image0][i][j] - (SubIb) / 2)
                if xbmin < 0:
                    xbmin = 0
                    SubIb_x = x0[Image0][i][j] * 2
                    # Position of the correlation matrix centroid:
                    xpeak_ac = (SubIb_x - SubIr) / 2
                    testexp = True
                else:
                    # Position of the correlation matrix centroid:
                    xpeak_ac = (SubIb - SubIr) / 2
                    testexp = False

                ybmin = int(y0[Image0][i][j] - (SubIb) / 2)
                if ybmin < 0:
                    ybmin = 0
                    SubIb_y = y0[Image0][i][j] * 2

                    ypeak_ac = (SubIb_y - SubIr) / 2
                    testeyp = True
                else:
                    ypeak_ac = (SubIb - SubIr) / 2
                    testeyp = False

                # Crop reference and search images according to SubIr and SubIb values:
                Ir = Iun[yrmin:yrmin + SubIr, xrmin:xrmin + SubIr]
                Ib = Id[ybmin:ybmin + SubIb, xbmin:xbmin + SubIb]

                try:
                    # Correlation:
                    c = cv.matchTemplate(Ib, Ir, Method)

                    # Arrange the correlation matrix into vector shape:
                    pp = np.arange(0, c.shape[0])

                    # Search for the maximum coefficient of correlation and its position:
                    _, maxVal, _, maxc = cv.minMaxLoc(c)

                    # If the maximum coefficient of correlation is lower than the correlation criterion, the node ij assumes the
                    # value nan, indicating the loss of correlation:
                    if maxVal < Criterion:

                        ui = float('nan')
                        vi = float('nan')

                    else:

                        # Position of the correlation peak:
                        [ypeak_cc, xpeak_cc] = np.unravel_index(c.argmax(), c.shape)
                        ypeak_c = pp[ypeak_cc]
                        xpeak_c = pp[xpeak_cc]

                        h = 2;
                        # Correlation matrix crop:
                        pc = c[ypeak_c - h:ypeak_c + h + 1, xpeak_c - h: xpeak_c + h + 1]

                        # Matrix 5x5 containing the maximum value:
                        xc = np.linspace(xpeak_c - h, xpeak_c + h, 5)
                        yc = np.linspace(ypeak_c - h, ypeak_c + h, 5)

                        # Matrix containing the maximum value (interpolation):
                        if (5 * OpiSub % 2) == 0:
                            factor = 5 * OpiSub + 1
                        else:
                            factor = 5 * OpiSub

                        xci = np.linspace(xpeak_c - h, xpeak_c + h, factor)
                        yci = np.linspace(ypeak_c - h, ypeak_c + h, factor)

                        # Performs the interpolation over a 2D domain using bicubic spline:
                        f = interp2d(yc, xc, pc, kind='cubic')

                        del c, pc

                        ccinterp = f(yci, xci)

                        # Search for the maximum interpolated coefficient of correlation and its position:
                        _, maxVal_interp, _, maxc_interp = cv.minMaxLoc(ccinterp)

                        # Interpolated position of the correlation peak:
                        xpeak = xci[maxc_interp[0]];
                        ypeak = yci[maxc_interp[1]]

                        # Displacement calculation:
                        ui = xpeak - xpeak_ac
                        vi = ypeak - ypeak_ac

                except:
                    ui = float('nan')
                    vi = float('nan')

                if Type == 'Lagrangian':

                    x0[Image1][i][j] = x0[Image0][i][j] + ui
                    y0[Image1][i][j] = y0[Image0][i][j] + vi
                    u0[Image1][i][j] = ui;
                    v0[Image1][i][j] = vi;

                else:

                    x0[Image1][i][j] = x0[Image0][i][j]
                    y0[Image1][i][j] = y0[Image0][i][j]
                    u0[Image1][i][j] = ui;
                    v0[Image1][i][j] = vi;

########################################################################################################################
# Function to load the ROI and subset points construction
########################################################################################################################
def SelectionLoad(menu, console, file_var, V, file, capturedFolder, SubIr, SubIb, Nx,
         Ny, Valpixel, Valmm, Opi, OpiSub,Version, TypeCut, NumCut, Adjust, Alpha, progression, progression_bar, canvas,
                  canvas_text, Method, Correlation, Criterion, Step, file_var_ROI, file_ROI, Interpolation, Filtering,
                  Kernel,ResultsName):

    global xriL, yriL, uriL, vriL, fileNamesSelection, selectionPath, resultsPath, fileNames, Format, Images
    global  xriL_mem, yriL_mem, uriL_mem, vriL_mem

    if Interpolation.get() == 'After':
        Opi.set(1)
        console.insert(tk.END,
                       f'The interpolation factor was changed to 1 according to the interpolation preference (after correlation)!\n\n')
        console.see('insert')

    file_load_mesh = filedialog.askopenfilename()

    file_ROI.set(file_load_mesh)

    l = open(file_load_mesh, "r")
    w = 2
    lines = l.readlines()
    selectionPath = lines[w].rstrip("\n"); w = w + 2

    save(menu, console, file_var, V, file, capturedFolder, SubIr, SubIb, Nx,
         Ny, Valpixel, Valmm, Opi, OpiSub, Version, TypeCut, NumCut, Adjust, Alpha, Method, Correlation, Criterion, Step, Interpolation, Filtering, Kernel)

    selectionPath = capturedFolder.get().rsplit('/', 1)[0] + '/Image_Selection'
    if not os.path.exists(selectionPath):
        os.makedirs(selectionPath)
        console.insert(tk.END, f'The Image_Selection folder was created\n\n')
        console.see('insert')
    else:
        console.insert(tk.END, 'The Image_Selection folder is already in the main directory\n\n')
        console.see('insert')

    for files in os.listdir(selectionPath):
        if files.endswith(Format):
            os.remove(os.path.join(selectionPath, files))

    resultsPath = capturedFolder.get().rsplit('/', 1)[0] + f'/{ResultsName.get()}'
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)
        console.insert(tk.END, f'The {ResultsName.get()} folder was created\n\n')
        console.see('insert')
    else:
        console.insert(tk.END, f'The {ResultsName.get()} folder is already in the main directory\n\n')
        console.see('insert')

    for files in os.listdir(resultsPath):
        if files.endswith(".dat"):
            os.remove(os.path.join(resultsPath, files))

    if Adjust.get() == 0.0:
        adjust = cv.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    else:
        adjust = cv.createCLAHE(clipLimit=Adjust.get(), tileGridSize=(8, 8))

    xinitline = int(lines[w]); w = w + 1
    yinitline = int(lines[w]); w = w + 1
    dxline = int(lines[w]); w = w + 1
    dyline = int(lines[w]); w = w + 2

    for k in range(1, Images + 1):
        green_length = int(756 * ((k) / Images))
        progression.coords(progression_bar, 0, 0, green_length, 25);
        progression.itemconfig(canvas_text, text=f'{k} of {Images} - {100 * (k) / Images:.2f}%')

        cv.imwrite(selectionPath + f'\\Image{k}' + Format,
                   adjust.apply(ndimage.rotate(cv.imread(fileNames[k - 1], 0), Alpha.get(), reshape=False)[
                                    int(yinitline):int(yinitline + dyline), int(xinitline):int(xinitline + dxline)]))

    fileNamesSelection = sorted(glob.glob(selectionPath + '/*'), key=stringToList)

    console.insert(tk.END, f'{Images} images were cropped and adjusted\n\n')
    console.see('insert')

    xinitline = int(lines[w]); w = w + 1
    yinitline = int(lines[w]); w = w + 1
    dxline = int(lines[w]); w = w + 1
    dyline = int(lines[w]); w = w + 2

    console.insert(tk.END, f'ROI construction was successfully loaded in {file_ROI.get()}\n\n')
    console.see('insert')

    if Nx.get() == 0 and Ny.get() == 0:
        Nx.set(int(abs(dxline) / Step.get()))
        Ny.set(int(abs(dyline) / Step.get()))

    xriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
    xriL = np.frombuffer(xriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)
    yriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
    yriL = np.frombuffer(yriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)
    uriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
    uriL = np.frombuffer(uriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)
    vriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
    vriL = np.frombuffer(vriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)

    for i in range(0, Ny.get() + 1):
        for j in range(0, Nx.get() + 1):
            xriL[0, i, j] = xinitline * Opi.get() + ((j) * dxline * Opi.get()) / (Nx.get())
            yriL[0, i, j] = yinitline * Opi.get() + ((i) * dyline * Opi.get()) / (Ny.get())

    if NumCut.get() != 0:
        if 'Rectangular' in TypeCut.get():
            for i in range(0, NumCut.get()):
                console.insert(tk.END, f'Select the rectangular cut region # {str(i + 1)} and press enter key\n\n')
                console.see('insert')
                image = cv.imread(fileNamesSelection[0])
                for i in range(0, Ny.get()):
                    for j in range(0, Nx.get()):
                        if np.isnan(xriL[0][i][int((Nx.get() + 1) / 2)]) or np.isnan(
                                xriL[0][i + 1][int((Nx.get() + 1) / 2)]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][i][int((Nx.get() + 1) / 2)] / Opi.get()),
                                                    int(yriL[0][i][int((Nx.get() + 1) / 2)] / Opi.get())), (
                                            int(xriL[0][i + 1][int((Nx.get() + 1) / 2)] / Opi.get()),
                                            int(yriL[0][i + 1][int((Nx.get() + 1) / 2)] / Opi.get())), (255, 0, 0), 3)
                        if np.isnan(xriL[0][int((Ny.get() + 1) / 2)][j]) or np.isnan(
                                xriL[0][int((Ny.get() + 1) / 2)][j + 1]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][int((Ny.get() + 1) / 2)][j] / Opi.get()),
                                                    int(yriL[0][int((Ny.get() + 1) / 2)][j] / Opi.get())), (
                                            int(xriL[0][int((Ny.get() + 1) / 2)][j + 1] / Opi.get()),
                                            int(yriL[0][int((Ny.get() + 1) / 2)][j + 1] / Opi.get())), (255, 0, 0), 3)

                for i in range(0, Ny.get() + 1):
                    for j in range(0, Nx.get()):
                        if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i][j + 1]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][i][j] / Opi.get()), int(yriL[0][i][j] / Opi.get())),
                                            (int(xriL[0][i][j + 1] / Opi.get()), int(yriL[0][i][j + 1] / Opi.get())),
                                            (0, 0, 255), 2)

                for i in range(0, Ny.get()):
                    for j in range(0, Nx.get() + 1):
                        if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i + 1][j]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][i][j] / Opi.get()), int(yriL[0][i][j] / Opi.get())),
                                            (int(xriL[0][i + 1][j] / Opi.get()), int(yriL[0][i + 1][j] / Opi.get())),
                                            (0, 0, 255), 2)

                dxfigure = int(image.shape[1])
                dyfigure = int(image.shape[0])

                screen_width = menu.winfo_screenwidth() - 100
                screen_height = menu.winfo_screenheight() - 100

                # Change size of displayed image:
                if dyfigure > screen_height:
                    ratio = screen_height / dyfigure
                    image_figure = cv.resize(image,
                                             (int(screen_height * dxfigure / dyfigure), screen_height))
                    # image_figure = cv.cvtColor(image_figure, cv.COLOR_GRAY2RGB)
                    xinitline, yinitline, dxline, dyline = SelectROI_DIC(menu, console,
                                                                         'Select the rectangular cut region',
                                                                         image_figure)

                    xinitline = xinitline / ratio
                    yinitline = yinitline / ratio
                    dxline = dxline / ratio
                    dyline = dyline / ratio
                else:
                    xinitline, yinitline, dxline, dyline = SelectROI_DIC(menu, console,
                                                                         'Select the rectangular cut region', image)

                cv.destroyAllWindows()

                for i in range(0, Ny.get() + 1):
                    for j in range(0, Nx.get() + 1):

                        if xinitline <= xriL[0][i][j] / Opi.get() <= xinitline + dxline and yinitline <= yriL[0][i][
                            j] / Opi.get() <= yinitline + dyline:
                            xriL[0][i][j] = float('nan')
                            yriL[0][i][j] = float('nan')

        if 'Circular inside' in TypeCut.get():
            for i in range(0, NumCut.get()):
                console.insert(tk.END, f'Select the circular cutting region # {str(i + 1)} and press enter key\n\n')
                console.see('insert')

                image = cv.imread(fileNamesSelection[0])
                for i in range(0, Ny.get()):
                    for j in range(0, Nx.get()):
                        if np.isnan(xriL[0][i][int((Nx.get() + 1) / 2)]) or np.isnan(
                                xriL[0][i + 1][int((Nx.get() + 1) / 2)]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][i][int((Nx.get() + 1) / 2)] / Opi.get()),
                                                    int(yriL[0][i][int((Nx.get() + 1) / 2)] / Opi.get())), (
                                            int(xriL[0][i + 1][int((Nx.get() + 1) / 2)] / Opi.get()),
                                            int(yriL[0][i + 1][int((Nx.get() + 1) / 2)] / Opi.get())), (255, 0, 0), 3)
                        if np.isnan(xriL[0][int((Ny.get() + 1) / 2)][j]) or np.isnan(
                                xriL[0][int((Ny.get() + 1) / 2)][j + 1]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][int((Ny.get() + 1) / 2)][j] / Opi.get()),
                                                    int(yriL[0][int((Ny.get() + 1) / 2)][j] / Opi.get())), (
                                            int(xriL[0][int((Ny.get() + 1) / 2)][j + 1] / Opi.get()),
                                            int(yriL[0][int((Ny.get() + 1) / 2)][j + 1] / Opi.get())), (255, 0, 0), 3)

                for i in range(0, Ny.get() + 1):
                    for j in range(0, Nx.get()):
                        if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i][j + 1]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][i][j] / Opi.get()), int(yriL[0][i][j] / Opi.get())),
                                            (int(xriL[0][i][j + 1] / Opi.get()), int(yriL[0][i][j + 1] / Opi.get())),
                                            (0, 0, 255), 2)

                for i in range(0, Ny.get()):
                    for j in range(0, Nx.get() + 1):
                        if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i + 1][j]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][i][j] / Opi.get()), int(yriL[0][i][j] / Opi.get())),
                                            (int(xriL[0][i + 1][j] / Opi.get()), int(yriL[0][i + 1][j] / Opi.get())),
                                            (0, 0, 255), 2)

                dxfigure = int(image.shape[1])
                dyfigure = int(image.shape[0])

                screen_width = menu.winfo_screenwidth() - 100
                screen_height = menu.winfo_screenheight() - 100

                # Change size of displayed image:
                if dyfigure > screen_height:
                    ratio = screen_height / dyfigure
                    image_figure = cv.resize(image,
                                             (int(screen_height * dxfigure / dyfigure), screen_height))
                    # image_figure = cv.cvtColor(image_figure, cv.COLOR_GRAY2RGB)
                    pointsCut = CircularROI_DIC(menu, console, 'Select the circular cut region', image_figure)

                    for w in range(0, len(pointsCut)):
                        pointsCut[w] = [x / ratio for x in pointsCut[w]]

                else:
                    pointsCut = CircularROI_DIC(menu, console, 'Select the circular cut region', image)

                cv.destroyAllWindows()

                polygon = Polygon(pointsCut)
                for i in range(0, Ny.get() + 1):
                    for j in range(0, Nx.get() + 1):
                        point = Point(xriL[0][i][j] / Opi.get(), yriL[0][i][j] / Opi.get())
                        if polygon.contains(point):
                            xriL[0][i][j] = float('nan')
                            yriL[0][i][j] = float('nan')
                            uriL[0][i][j] = float('nan')
                            vriL[0][i][j] = float('nan')

        if 'Circular outside' in TypeCut.get():
            for i in range(0, NumCut.get()):
                console.insert(tk.END, f'Select the circular cutting region # {str(i + 1)} and press enter key\n\n')
                console.see('insert')

                image = cv.imread(fileNamesSelection[0])
                for i in range(0, Ny.get()):
                    for j in range(0, Nx.get()):
                        if np.isnan(xriL[0][i][int((Nx.get() + 1) / 2)]) or np.isnan(
                                xriL[0][i + 1][int((Nx.get() + 1) / 2)]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][i][int((Nx.get() + 1) / 2)] / Opi.get()),
                                                    int(yriL[0][i][int((Nx.get() + 1) / 2)] / Opi.get())), (
                                            int(xriL[0][i + 1][int((Nx.get() + 1) / 2)] / Opi.get()),
                                            int(yriL[0][i + 1][int((Nx.get() + 1) / 2)] / Opi.get())), (255, 0, 0), 3)
                        if np.isnan(xriL[0][int((Ny.get() + 1) / 2)][j]) or np.isnan(
                                xriL[0][int((Ny.get() + 1) / 2)][j + 1]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][int((Ny.get() + 1) / 2)][j] / Opi.get()),
                                                    int(yriL[0][int((Ny.get() + 1) / 2)][j] / Opi.get())), (
                                            int(xriL[0][int((Ny.get() + 1) / 2)][j + 1] / Opi.get()),
                                            int(yriL[0][int((Ny.get() + 1) / 2)][j + 1] / Opi.get())), (255, 0, 0), 3)

                for i in range(0, Ny.get() + 1):
                    for j in range(0, Nx.get()):
                        if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i][j + 1]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][i][j] / Opi.get()), int(yriL[0][i][j] / Opi.get())),
                                            (int(xriL[0][i][j + 1] / Opi.get()), int(yriL[0][i][j + 1] / Opi.get())),
                                            (0, 0, 255), 2)

                for i in range(0, Ny.get()):
                    for j in range(0, Nx.get() + 1):
                        if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i + 1][j]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][i][j] / Opi.get()), int(yriL[0][i][j] / Opi.get())),
                                            (int(xriL[0][i + 1][j] / Opi.get()), int(yriL[0][i + 1][j] / Opi.get())),
                                            (0, 0, 255), 2)

                dxfigure = int(image.shape[1])
                dyfigure = int(image.shape[0])

                screen_width = menu.winfo_screenwidth() - 100
                screen_height = menu.winfo_screenheight() - 100

                # Change size of displayed image:
                if dyfigure > screen_height:
                    ratio = screen_height / dyfigure
                    image_figure = cv.resize(image,
                                             (int(screen_height * dxfigure / dyfigure), screen_height))
                    # image_figure = cv.cvtColor(image_figure, cv.COLOR_GRAY2RGB)
                    pointsCut = CircularROI_DIC(menu, console, 'Select the circular cut region', image_figure)

                    for w in range(0, len(pointsCut)):
                        pointsCut[w] = [x / ratio for x in pointsCut[w]]

                else:
                    pointsCut = CircularROI_DIC(menu, console, 'Select the circular cut region', image)

                cv.destroyAllWindows()

                polygon = Polygon(pointsCut)
                for i in range(0, Ny.get() + 1):
                    for j in range(0, Nx.get() + 1):
                        point = Point(xriL[0][i][j] / Opi.get(), yriL[0][i][j] / Opi.get())
                        if polygon.contains(point):
                            pass
                        else:
                            xriL[0][i][j] = float('nan')
                            yriL[0][i][j] = float('nan')
                            uriL[0][i][j] = float('nan')
                            vriL[0][i][j] = float('nan')

        if 'Free' in TypeCut.get():
            for i in range(0, NumCut.get()):
                console.insert(tk.END, f'Select the freehand cutting region # {str(i + 1)} and press enter key\n\n')
                console.see('insert')

                image = cv.imread(fileNamesSelection[0])
                for i in range(0, Ny.get()):
                    for j in range(0, Nx.get()):
                        if np.isnan(xriL[0][i][int((Nx.get() + 1) / 2)]) or np.isnan(
                                xriL[0][i + 1][int((Nx.get() + 1) / 2)]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][i][int((Nx.get() + 1) / 2)] / Opi.get()),
                                                    int(yriL[0][i][int((Nx.get() + 1) / 2)] / Opi.get())), (
                                            int(xriL[0][i + 1][int((Nx.get() + 1) / 2)] / Opi.get()),
                                            int(yriL[0][i + 1][int((Nx.get() + 1) / 2)] / Opi.get())), (255, 0, 0), 3)
                        if np.isnan(xriL[0][int((Ny.get() + 1) / 2)][j]) or np.isnan(
                                xriL[0][int((Ny.get() + 1) / 2)][j + 1]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][int((Ny.get() + 1) / 2)][j] / Opi.get()),
                                                    int(yriL[0][int((Ny.get() + 1) / 2)][j] / Opi.get())), (
                                            int(xriL[0][int((Ny.get() + 1) / 2)][j + 1] / Opi.get()),
                                            int(yriL[0][int((Ny.get() + 1) / 2)][j + 1] / Opi.get())), (255, 0, 0), 3)

                for i in range(0, Ny.get() + 1):
                    for j in range(0, Nx.get()):
                        if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i][j + 1]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][i][j] / Opi.get()), int(yriL[0][i][j] / Opi.get())),
                                            (int(xriL[0][i][j + 1] / Opi.get()), int(yriL[0][i][j + 1] / Opi.get())),
                                            (0, 0, 255), 2)

                for i in range(0, Ny.get()):
                    for j in range(0, Nx.get() + 1):
                        if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i + 1][j]):
                            pass
                        else:
                            image = cv.line(image, (int(xriL[0][i][j] / Opi.get()), int(yriL[0][i][j] / Opi.get())),
                                            (int(xriL[0][i + 1][j] / Opi.get()), int(yriL[0][i + 1][j] / Opi.get())),
                                            (0, 0, 255), 2)

                dxfigure = int(image.shape[1])
                dyfigure = int(image.shape[0])

                screen_width = menu.winfo_screenwidth() - 100
                screen_height = menu.winfo_screenheight() - 100

                # Change size of displayed image:
                if dyfigure > screen_height:
                    ratio = screen_height / dyfigure
                    image_figure = cv.resize(image,
                                             (int(screen_height * dxfigure / dyfigure), screen_height))
                    # image_figure = cv.cvtColor(image_figure, cv.COLOR_GRAY2RGB)
                    pointsCut = freehandCut('Select the freehand cutting region', image_figure)

                    for w in range(0, len(pointsCut)):
                        pointsCut[w] = [x / ratio for x in pointsCut[w]]

                else:
                    pointsCut = freehandCut('Select the freehand cutting region', image)

                cv.destroyAllWindows()

                polygon = Polygon(pointsCut)
                for i in range(0, Ny.get() + 1):
                    for j in range(0, Nx.get() + 1):
                        point = Point(xriL[0][i][j] / Opi.get(), yriL[0][i][j] / Opi.get())
                        if polygon.contains(point):
                            xriL[0][i][j] = float('nan')
                            yriL[0][i][j] = float('nan')
                            uriL[0][i][j] = float('nan')
                            vriL[0][i][j] = float('nan')

    fig = plt.figure()
    ax = fig.gca()
    dxplot = int(cv.imread(fileNamesSelection[0]).shape[1])
    dyplot = int(cv.imread(fileNamesSelection[0]).shape[0])

    ratio_plot = dxplot / dyplot;

    if ratio_plot <= 1.33333333:

        dxplotdark = dyplot * 1.33333333
        dyplotdark = dyplot
        ax.imshow(cv.imread(fileNamesSelection[0]), zorder=2)
        ax.plot(np.transpose(xriL[0][:][:] / Opi.get()), np.transpose(yriL[0][:][:] / Opi.get()), color='red',
                linewidth=1, zorder=3)
        ax.plot(xriL[0][:][:] / Opi.get(), yriL[0][:][:] / Opi.get(), color='red', linewidth=1, zorder=3)

        winSub = False
        for i in range(0, Ny.get() + 1):
            for j in range(0, Nx.get() + 1):
                if ~np.isnan(xriL[0][i][j]):

                    ax.plot(
                        [xriL[0][i][j] / Opi.get() - SubIr.get() / 2, xriL[0][i][j] / Opi.get() + SubIr.get() / 2,
                         xriL[0][i][j] / Opi.get() + SubIr.get() / 2, xriL[0][i][j] / Opi.get() - SubIr.get() / 2,
                         xriL[0][i][j] / Opi.get() - SubIr.get() / 2],
                        [yriL[0][i][j] / Opi.get() + SubIr.get() / 2, yriL[0][i][j] / Opi.get() + SubIr.get() / 2,
                         yriL[0][i][j] / Opi.get() - SubIr.get() / 2, yriL[0][i][j] / Opi.get() - SubIr.get() / 2,
                         yriL[0][i][j] / Opi.get() + SubIr.get() / 2],
                        color='#00FF00', linewidth=1, zorder=3)  # SubIr window

                    ax.plot(
                        [xriL[0][i][j] / Opi.get() - SubIb.get() / 2, xriL[0][i][j] / Opi.get() + SubIb.get() / 2,
                         xriL[0][i][j] / Opi.get() + SubIb.get() / 2, xriL[0][i][j] / Opi.get() - SubIb.get() / 2,
                         xriL[0][i][j] / Opi.get() - SubIb.get() / 2],
                        [yriL[0][i][j] / Opi.get() + SubIb.get() / 2, yriL[0][i][j] / Opi.get() + SubIb.get() / 2,
                         yriL[0][i][j] / Opi.get() - SubIb.get() / 2, yriL[0][i][j] / Opi.get() - SubIb.get() / 2,
                         yriL[0][i][j] / Opi.get() + SubIb.get() / 2],
                        color='blue', linewidth=1, zorder=3)  # SubIb window

                    winSub = True
                    break
                else:
                    continue

            if winSub: break

        ax.plot([0 - dxplotdark / 2 + dxplot / 2, 0 - dxplotdark / 2 + dxplot / 2,
                 dxplotdark - dxplotdark / 2 + dxplot / 2, dxplotdark - dxplotdark / 2 + dxplot / 2,
                 0 - dxplotdark / 2 + dxplot / 2], [0, dyplotdark, dyplotdark, 0, 0], 'black', zorder=1)
        ax.fill_between([0 - dxplotdark / 2 + dxplot / 2, 0 - dxplotdark / 2 + dxplot / 2,
                         dxplotdark - dxplotdark / 2 + dxplot / 2, dxplotdark - dxplotdark / 2 + dxplot / 2,
                         0 - dxplotdark / 2 + dxplot / 2], [0, dyplotdark, dyplotdark, 0, 0], color='black',
                        zorder=1)
        ax.axis('off')
        plt.xlim(-dxplotdark / 2 + dxplot / 2, dxplotdark - dxplotdark / 2 + dxplot / 2)
        plt.ylim(0, dyplotdark)
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    else:
        dxplotdark = dxplot
        dyplotdark = dxplot / 1.33333333
        ax.imshow(cv.imread(fileNamesSelection[0]), zorder=2)
        ax.plot(np.transpose(xriL[0][:][:] / Opi.get()), np.transpose(yriL[0][:][:] / Opi.get()), color='red',
                linewidth=1, zorder=3)
        ax.plot(xriL[0][:][:] / Opi.get(), yriL[0][:][:] / Opi.get(), color='red', linewidth=1, zorder=3)

        winSub = False
        for i in range(0, Ny.get() + 1):
            for j in range(0, Nx.get() + 1):
                if ~np.isnan(xriL[0][i][j]):

                    ax.plot(
                        [xriL[0][i][j] / Opi.get() - SubIr.get() / 2, xriL[0][i][j] / Opi.get() + SubIr.get() / 2,
                         xriL[0][i][j] / Opi.get() + SubIr.get() / 2, xriL[0][i][j] / Opi.get() - SubIr.get() / 2,
                         xriL[0][i][j] / Opi.get() - SubIr.get() / 2],
                        [yriL[0][i][j] / Opi.get() + SubIr.get() / 2, yriL[0][i][j] / Opi.get() + SubIr.get() / 2,
                         yriL[0][i][j] / Opi.get() - SubIr.get() / 2, yriL[0][i][j] / Opi.get() - SubIr.get() / 2,
                         yriL[0][i][j] / Opi.get() + SubIr.get() / 2],
                        color='#00FF00', linewidth=1, zorder=3)  # SubIr window

                    ax.plot(
                        [xriL[0][i][j] / Opi.get() - SubIb.get() / 2, xriL[0][i][j] / Opi.get() + SubIb.get() / 2,
                         xriL[0][i][j] / Opi.get() + SubIb.get() / 2, xriL[0][i][j] / Opi.get() - SubIb.get() / 2,
                         xriL[0][i][j] / Opi.get() - SubIb.get() / 2],
                        [yriL[0][i][j] / Opi.get() + SubIb.get() / 2, yriL[0][i][j] / Opi.get() + SubIb.get() / 2,
                         yriL[0][i][j] / Opi.get() - SubIb.get() / 2, yriL[0][i][j] / Opi.get() - SubIb.get() / 2,
                         yriL[0][i][j] / Opi.get() + SubIb.get() / 2],
                        color='blue', linewidth=1, zorder=3)  # SubIb window

                    winSub = True
                    break
                else:
                    continue

            if winSub: break

        ax.plot([0, 0, dxplotdark, dxplotdark, 0],
                [0 - dyplotdark / 2 + dyplot / 2, +dyplotdark - dyplotdark / 2 + dyplot / 2,
                 +dyplotdark - dyplotdark / 2 + dyplot / 2, 0 - dyplotdark / 2 + dyplot / 2,
                 0 - dyplotdark / 2 + dyplot / 2], 'black', zorder=1)
        ax.fill_between([0, 0, dxplotdark, dxplotdark, 0],
                        [0 - dyplotdark / 2 + dyplot / 2, +dyplotdark - dyplotdark / 2 + dyplot / 2,
                         +dyplotdark - dyplotdark / 2 + dyplot / 2, 0 - dyplotdark / 2 + dyplot / 2,
                         0 - dyplotdark / 2 + dyplot / 2], color='black', zorder=1)
        ax.axis('off')
        plt.xlim(0, dxplotdark)
        plt.ylim(-dyplotdark / 2 + dyplot / 2, dyplotdark - dyplotdark / 2 + dyplot / 2)
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    image = np.flip(image.reshape((h, w, 3)), axis=0)

    plt.cla()
    plt.clf()

    image = cv.resize(image, (640, 480))
    cv.putText(image, f'CORRELATION MESH - {np.count_nonzero(~np.isnan(xriL[0][:][:]))} nodes', (20, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
    cv.putText(image, 'REFERENCE SUBSET', (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
    cv.putText(image, 'SEARCH SUBSET', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

    image = ImageTk.PhotoImage(Image.fromarray(image))

    canvas.image = image
    canvas.configure(image=image)

    console.insert(tk.END, 'Start the correlation process\n\n')
    console.see('insert')

    messagebox.showinfo('Information', 'Press start to initialize the correlation!')

########################################################################################################################
# Save subset points
########################################################################################################################
def SaveSubsets(console):
    global xriL, yriL, uriL, vriL

    console.insert(tk.END,
                   '###########################################################################################\n\n')
    console.insert(tk.END, 'Select the folder to save the position of subsets - .csv format\n\n')
    console.see('insert')

    mesh_folder = filedialog.askdirectory()

    np.savetxt(f'{mesh_folder}\\x_img.csv', xriL[:][:][0], delimiter='\t')
    np.savetxt(f'{mesh_folder}\\y_img.csv', yriL[:][:][0], delimiter='\t')
    np.savetxt(f'{mesh_folder}\\u_img.csv', uriL[:][:][0], delimiter='\t')
    np.savetxt(f'{mesh_folder}\\v_img.csv', vriL[:][:][0], delimiter='\t')

    messagebox.showinfo('Information', 'The position of subsets has been successfully saved!')

########################################################################################################################
# Load subset points
########################################################################################################################
def LoadSubsets(console, canvas, capturedFolder, Nx, Ny, SubIr, SubIb, Interpolation, Opi, ResultsName):
    global xriL, yriL, uriL, vriL, fileNamesSelection, selectionPath, resultsPath, Format, Images
    global xriL_mem, yriL_mem, uriL_mem, vriL_mem

    console.insert(tk.END,
                   '###########################################################################################\n\n')
    console.see('insert')

    if Interpolation.get() == 'After':
        Opi.set(1)
        console.insert(tk.END,f'The interpolation factor was changed to 1 according to the interpolation preference (after correlation)!\n\n')
        console.see('insert')

    selectionPath = capturedFolder.get().rsplit('/', 1)[0] + '/Image_Selection'

    fileNamesSelection = sorted(glob.glob(selectionPath + '/*'), key=stringToList)

    resultsPath = capturedFolder.get().rsplit('/', 1)[0] + f'/{ResultsName.get()}'
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)
        console.insert(tk.END, f'The {ResultsName.get()} folder was created\n\n')
        console.see('insert')
    else:
        console.insert(tk.END, f'The {ResultsName.get()} folder is already in the main directory\n\n')
        console.see('insert')

    for file in os.listdir(resultsPath):
        if file.endswith(".dat"):
            os.remove(os.path.join(resultsPath, file))

    mesh_folder = filedialog.askdirectory()

    subset_test = genfromtxt(f'{mesh_folder}\\x_img.csv', delimiter='\t')

    ans = messagebox.askquestion('Subset construction', 'The file was generated by iCorrVision-2D software?',
                                 icon='question')
    if ans == 'no':
        Ny.set(subset_test.shape[0] - 1)
        Nx.set(subset_test.shape[1] - 2)

        xriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
        xriL = np.frombuffer(xriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)
        yriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
        yriL = np.frombuffer(yriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)
        uriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
        uriL = np.frombuffer(uriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)
        vriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
        vriL = np.frombuffer(vriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)

        xriL[:][:][0] = genfromtxt(f'{mesh_folder}\\x_img.csv', delimiter='\t')[0:Ny.get() + 1, 0:Nx.get() + 1]
        yriL[:][:][0] = genfromtxt(f'{mesh_folder}\\y_img.csv', delimiter='\t')[0:Ny.get() + 1, 0:Nx.get() + 1]
        uriL[:][:][0] = genfromtxt(f'{mesh_folder}\\u_img.csv', delimiter='\t')[0:Ny.get() + 1, 0:Nx.get() + 1]
        vriL[:][:][0] = genfromtxt(f'{mesh_folder}\\v_img.csv', delimiter='\t')[0:Ny.get() + 1, 0:Nx.get() + 1]

        xriL[:][:][0] = xriL[:][:][0] * Opi.get()
        yriL[:][:][0] = yriL[:][:][0] * Opi.get()
        uriL[:][:][0] = uriL[:][:][0] * Opi.get()
        vriL[:][:][0] = vriL[:][:][0] * Opi.get()

    else:
        Ny.set(subset_test.shape[0] - 1)
        Nx.set(subset_test.shape[1] - 1)

        xriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
        xriL = np.frombuffer(xriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)
        yriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
        yriL = np.frombuffer(yriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)
        uriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
        uriL = np.frombuffer(uriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)
        vriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
        vriL = np.frombuffer(vriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)

        xriL[:][:][0] = genfromtxt(f'{mesh_folder}\\x_img.csv', delimiter='\t')[0:Ny.get() + 1, 0:Nx.get() + 1]
        yriL[:][:][0] = genfromtxt(f'{mesh_folder}\\y_img.csv', delimiter='\t')[0:Ny.get() + 1, 0:Nx.get() + 1]
        uriL[:][:][0] = genfromtxt(f'{mesh_folder}\\u_img.csv', delimiter='\t')[0:Ny.get() + 1, 0:Nx.get() + 1]
        vriL[:][:][0] = genfromtxt(f'{mesh_folder}\\v_img.csv', delimiter='\t')[0:Ny.get() + 1, 0:Nx.get() + 1]

    fig = plt.figure()
    ax = fig.gca()
    dxplot = int(cv.imread(fileNamesSelection[0]).shape[1])
    dyplot = int(cv.imread(fileNamesSelection[0]).shape[0])

    ratio_plot = dxplot / dyplot;

    if ratio_plot <= 1.33333333:

        dxplotdark = dyplot * 1.33333333
        dyplotdark = dyplot
        ax.imshow(cv.imread(fileNamesSelection[0]), zorder=2)
        ax.plot(np.transpose(xriL[0][:][:] / Opi.get()), np.transpose(yriL[0][:][:] / Opi.get()), color='red',
                linewidth=1, zorder=3)
        ax.plot(xriL[0][:][:] / Opi.get(), yriL[0][:][:] / Opi.get(), color='red', linewidth=1, zorder=3)

        winSub = False
        for i in range(0, Ny.get() + 1):
            for j in range(0, Nx.get() + 1):
                if ~np.isnan(xriL[0][i][j]):

                    ax.plot(
                        [xriL[0][i][j] / Opi.get() - SubIr.get() / 2, xriL[0][i][j] / Opi.get() + SubIr.get() / 2,
                         xriL[0][i][j] / Opi.get() + SubIr.get() / 2, xriL[0][i][j] / Opi.get() - SubIr.get() / 2,
                         xriL[0][i][j] / Opi.get() - SubIr.get() / 2],
                        [yriL[0][i][j] / Opi.get() + SubIr.get() / 2, yriL[0][i][j] / Opi.get() + SubIr.get() / 2,
                         yriL[0][i][j] / Opi.get() - SubIr.get() / 2, yriL[0][i][j] / Opi.get() - SubIr.get() / 2,
                         yriL[0][i][j] / Opi.get() + SubIr.get() / 2],
                        color='#00FF00', linewidth=1, zorder=3)  # SubIr window

                    ax.plot(
                        [xriL[0][i][j] / Opi.get() - SubIb.get() / 2, xriL[0][i][j] / Opi.get() + SubIb.get() / 2,
                         xriL[0][i][j] / Opi.get() + SubIb.get() / 2, xriL[0][i][j] / Opi.get() - SubIb.get() / 2,
                         xriL[0][i][j] / Opi.get() - SubIb.get() / 2],
                        [yriL[0][i][j] / Opi.get() + SubIb.get() / 2, yriL[0][i][j] / Opi.get() + SubIb.get() / 2,
                         yriL[0][i][j] / Opi.get() - SubIb.get() / 2, yriL[0][i][j] / Opi.get() - SubIb.get() / 2,
                         yriL[0][i][j] / Opi.get() + SubIb.get() / 2],
                        color='blue', linewidth=1, zorder=3)  # SubIb window

                    winSub = True
                    break
                else:
                    continue

            if winSub: break

        ax.plot([0 - dxplotdark / 2 + dxplot / 2, 0 - dxplotdark / 2 + dxplot / 2,
                 dxplotdark - dxplotdark / 2 + dxplot / 2, dxplotdark - dxplotdark / 2 + dxplot / 2,
                 0 - dxplotdark / 2 + dxplot / 2], [0, dyplotdark, dyplotdark, 0, 0], 'black', zorder=1)
        ax.fill_between([0 - dxplotdark / 2 + dxplot / 2, 0 - dxplotdark / 2 + dxplot / 2,
                         dxplotdark - dxplotdark / 2 + dxplot / 2, dxplotdark - dxplotdark / 2 + dxplot / 2,
                         0 - dxplotdark / 2 + dxplot / 2], [0, dyplotdark, dyplotdark, 0, 0], color='black',
                        zorder=1)
        ax.axis('off')
        plt.xlim(-dxplotdark / 2 + dxplot / 2, dxplotdark - dxplotdark / 2 + dxplot / 2)
        plt.ylim(0, dyplotdark)
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    else:
        dxplotdark = dxplot
        dyplotdark = dxplot / 1.33333333
        ax.imshow(cv.imread(fileNamesSelection[0]), zorder=2)
        ax.plot(np.transpose(xriL[0][:][:] / Opi.get()), np.transpose(yriL[0][:][:] / Opi.get()), color='red',
                linewidth=1, zorder=3)
        ax.plot(xriL[0][:][:] / Opi.get(), yriL[0][:][:] / Opi.get(), color='red', linewidth=1, zorder=3)

        winSub = False
        for i in range(0, Ny.get() + 1):
            for j in range(0, Nx.get() + 1):
                if ~np.isnan(xriL[0][i][j]):

                    ax.plot(
                        [xriL[0][i][j] / Opi.get() - SubIr.get() / 2, xriL[0][i][j] / Opi.get() + SubIr.get() / 2,
                         xriL[0][i][j] / Opi.get() + SubIr.get() / 2, xriL[0][i][j] / Opi.get() - SubIr.get() / 2,
                         xriL[0][i][j] / Opi.get() - SubIr.get() / 2],
                        [yriL[0][i][j] / Opi.get() + SubIr.get() / 2, yriL[0][i][j] / Opi.get() + SubIr.get() / 2,
                         yriL[0][i][j] / Opi.get() - SubIr.get() / 2, yriL[0][i][j] / Opi.get() - SubIr.get() / 2,
                         yriL[0][i][j] / Opi.get() + SubIr.get() / 2],
                        color='#00FF00', linewidth=1, zorder=3)  # SubIr window

                    ax.plot(
                        [xriL[0][i][j] / Opi.get() - SubIb.get() / 2, xriL[0][i][j] / Opi.get() + SubIb.get() / 2,
                         xriL[0][i][j] / Opi.get() + SubIb.get() / 2, xriL[0][i][j] / Opi.get() - SubIb.get() / 2,
                         xriL[0][i][j] / Opi.get() - SubIb.get() / 2],
                        [yriL[0][i][j] / Opi.get() + SubIb.get() / 2, yriL[0][i][j] / Opi.get() + SubIb.get() / 2,
                         yriL[0][i][j] / Opi.get() - SubIb.get() / 2, yriL[0][i][j] / Opi.get() - SubIb.get() / 2,
                         yriL[0][i][j] / Opi.get() + SubIb.get() / 2],
                        color='blue', linewidth=1, zorder=3)  # SubIb window

                    winSub = True
                    break
                else:
                    continue

            if winSub: break

        ax.plot([0, 0, dxplotdark, dxplotdark, 0],
                [0 - dyplotdark / 2 + dyplot / 2, +dyplotdark - dyplotdark / 2 + dyplot / 2,
                 +dyplotdark - dyplotdark / 2 + dyplot / 2, 0 - dyplotdark / 2 + dyplot / 2,
                 0 - dyplotdark / 2 + dyplot / 2], 'black', zorder=1)
        ax.fill_between([0, 0, dxplotdark, dxplotdark, 0],
                        [0 - dyplotdark / 2 + dyplot / 2, +dyplotdark - dyplotdark / 2 + dyplot / 2,
                         +dyplotdark - dyplotdark / 2 + dyplot / 2, 0 - dyplotdark / 2 + dyplot / 2,
                         0 - dyplotdark / 2 + dyplot / 2], color='black', zorder=1)
        ax.axis('off')
        plt.xlim(0, dxplotdark)
        plt.ylim(-dyplotdark / 2 + dyplot / 2, dyplotdark - dyplotdark / 2 + dyplot / 2)
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    image = np.flip(image.reshape((h, w, 3)), axis=0)

    plt.cla()
    plt.clf()


    image = cv.resize(image, (640, 480))
    cv.putText(image, f'CORRELATION MESH - {np.count_nonzero(~np.isnan(xriL[0][:][:]))} nodes', (20, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
    cv.putText(image, 'REFERENCE SUBSET', (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
    cv.putText(image, 'SEARCH SUBSET', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

    image = ImageTk.PhotoImage(Image.fromarray(image))

    canvas.image = image
    canvas.configure(image=image)

    console.insert(tk.END, 'Start the correlation process\n\n')
    console.see('insert')

    messagebox.showinfo('Information', 'Press start to initialize the correlation!')

########################################################################################################################
# Function to select the ROI and subset points construction
########################################################################################################################
def SelectionImage(menu, console, file_var, V, file, capturedFolder, SubIr, SubIb, Nx,
         Ny, Valpixel, Valmm, Opi, OpiSub, Version, TypeCut, NumCut, Adjust, Alpha, progression, progression_bar, canvas,
                   canvas_text, Method, Correlation, Criterion, Step, file_var_ROI, file_ROI, Interpolation, Filtering,
                   Kernel, ResultsName):

    global xriL, yriL, uriL, vriL, fileNamesSelection, selectionPath, resultsPath, fileNames, Format, Images
    global xriL_mem, yriL_mem, uriL_mem, vriL_mem

    try:
        fileNames
    except NameError:
        messagebox.showerror('Error','Please load project or select the image captured folder and indicate all parametrs before starting the selection process!')
    else:

        if Interpolation.get() == 'After':
            Opi.set(1)
            console.insert(tk.END, f'The interpolation factor was changed to 1 according to the interpolation preference (after correlation)!\n\n')
            console.see('insert')

        save(menu, console, file_var, V, file, capturedFolder, SubIr, SubIb, Nx,
         Ny, Valpixel, Valmm, Opi, OpiSub, Version, TypeCut, NumCut, Adjust, Alpha, Method, Correlation, Criterion, Step, Interpolation, Filtering, Kernel)

        # Creation of the Image_Selection folder:
        selectionPath = capturedFolder.get().rsplit('/', 1)[0]+'/Image_Selection'
        if not os.path.exists(selectionPath):
            os.makedirs(selectionPath)
            console.insert(tk.END, f'The Image_Selection folder was created\n\n')
            console.see('insert')
        else:
            console.insert(tk.END, 'The Image_Selection folder is already in the main directory\n\n')
            console.see('insert')

        for files in os.listdir(selectionPath):
            if files.endswith(Format):
                os.remove(os.path.join(selectionPath, files))

        # Creation of the Results_Correlation folder:
        resultsPath = capturedFolder.get().rsplit('/', 1)[0]+f'/{ResultsName.get()}'
        if not os.path.exists(resultsPath):
            os.makedirs(resultsPath)
            console.insert(tk.END, f'The {ResultsName.get()} folder was created\n\n')
            console.see('insert')
        else:
            console.insert(tk.END, f'The {ResultsName.get()} folder is already in the main directory\n\n')
            console.see('insert')

        for files in os.listdir(resultsPath):
            if files.endswith(".dat"):
                os.remove(os.path.join(resultsPath, files))

        # Save ROI log:
        file_var_ROI.set(True)

        console.insert(tk.END, 'Select a .dat file to save the region of interest (ROI)\n\n')
        console.see('insert')

        file_ROI.set(filedialog.asksaveasfilename())

        f = open(file_ROI.get(), "w+")

        f.write('iCorrVision-2D Correlation Module - ROI - ' + str(
            datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S")))
        f.write('\nImage selection folder:\n')
        f.write(str(selectionPath.rstrip("\n")))

        console.insert(tk.END, 'Select the region of interest (ROI) and press enter key\n\n')
        console.see('insert')

        if Adjust.get() == 0.0:
            adjust = cv.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
        else:
            adjust = cv.createCLAHE(clipLimit=Adjust.get(), tileGridSize=(8,8))

        dxfigure = int(cv.imread(fileNames[0]).shape[1])
        dyfigure = int(cv.imread(fileNames[0]).shape[0])

        screen_width = menu.winfo_screenwidth()-100
        screen_height = menu.winfo_screenheight()-100

        # Change size of displayed image:
        if dyfigure > screen_height:
            ratio = screen_height/dyfigure
            image_figure = adjust.apply(cv.resize(cv.imread(fileNames[0],0), (int(screen_height*dxfigure/dyfigure), screen_height)))
            image_figure = cv.cvtColor(image_figure,cv.COLOR_GRAY2RGB)
            xinitline,yinitline,dxline,dyline = SelectROI_DIC(menu, console,'Select the region of interest (ROI)',ndimage.rotate(image_figure,Alpha.get(), reshape=False))
            xinitline=xinitline/ratio
            yinitline=yinitline/ratio
            dxline=dxline/ratio
            dyline=dyline/ratio
        else:
            image_figure = adjust.apply(cv.imread(fileNames[0],0))
            image_figure = cv.cvtColor(image_figure,cv.COLOR_GRAY2RGB)
            xinitline,yinitline,dxline,dyline = SelectROI_DIC(menu, console,'Select the region of interest (ROI)',ndimage.rotate(image_figure,Alpha.get(), reshape=False))

        cv.destroyAllWindows()

        f.write('\nROI:\n')
        f.write(str(int(xinitline)) + '\n')
        f.write(str(int(yinitline)) + '\n')
        f.write(str(int(dxline)) + '\n')
        f.write(str(int(dyline)))

        for k in range(1,Images+1):
            green_length = int(756*((k)/Images))
            progression.coords(progression_bar, 0, 0, green_length, 25); progression.itemconfig(canvas_text, text=f'{k} of {Images} - {100*(k)/Images:.2f}%')

            cv.imwrite(selectionPath + f'\\Image{k}' + Format,
                       adjust.apply(ndimage.rotate(cv.imread(fileNames[k - 1], 0), Alpha.get(), reshape=False)[
                                    int(yinitline):int(yinitline + dyline), int(xinitline):int(xinitline + dxline)]))

        fileNamesSelection = sorted(glob.glob(selectionPath+'/*'),key=stringToList)

        console.insert(tk.END, f'{Images} images were cropped and adjusted\n\n')
        console.insert(tk.END, 'Select the region for the subset construction and press enter key\n\n')
        console.see('insert')

        dxfigure = int(cv.imread(fileNamesSelection[0]).shape[1])
        dyfigure = int(cv.imread(fileNamesSelection[0]).shape[0])

        screen_width = menu.winfo_screenwidth() - 100
        screen_height = menu.winfo_screenheight() - 100

        # Change size of displayed image:
        if dyfigure > screen_height:
            ratio = screen_height / dyfigure
            if Version.get() == 'Eulerian':
                image_figure = cv.resize(cv.imread(fileNamesSelection[-1], 0),
                                         (int(screen_height * dxfigure / dyfigure), screen_height))
                image_figure = cv.cvtColor(image_figure, cv.COLOR_GRAY2RGB)
                xinitline, yinitline, dxline, dyline = SelectROI_DIC(menu, console,'Select the region for the subset construction on last captured image (Eulerian)',
                                                                    image_figure)
            else:
                image_figure = cv.resize(cv.imread(fileNamesSelection[0], 0),
                                         (int(screen_height * dxfigure / dyfigure), screen_height))
                image_figure = cv.cvtColor(image_figure, cv.COLOR_GRAY2RGB)
                xinitline,yinitline,dxline,dyline = SelectROI_DIC(menu, console,'Select the region for the subset construction on first captured image (Lagrangian)',
                                                                 image_figure)
            xinitline = xinitline / ratio
            yinitline = yinitline / ratio
            dxline = dxline / ratio
            dyline = dyline / ratio
        else:
            if Version.get() == 'Eulerian':
                xinitline,yinitline,dxline,dyline = SelectROI_DIC(menu, console,'Select the region for the subset construction on last captured image (Eulerian)',
                                                                 cv.imread(fileNamesSelection[-1]))
            else:
                xinitline, yinitline, dxline, dyline = SelectROI_DIC(menu, console,'Select the region for the subset construction on first captured image (Lagrangian)',
                                                                    cv.imread(fileNamesSelection[0]))
        cv.destroyAllWindows()

        f.write('\nRegion for the subset construction:\n')
        f.write(str(int(xinitline)) + '\n')
        f.write(str(int(yinitline)) + '\n')
        f.write(str(int(dxline)) + '\n')
        f.write(str(int(dyline)))
        f.close()

        console.insert(tk.END, f'The ROI and subset construction was successfully saved in {file_ROI.get()}\n\n')
        console.see('insert')

        if Nx.get() == 0 and Ny.get() == 0:
            Nx.set(int(abs(dxline)/Step.get()))
            Ny.set(int(abs(dyline)/Step.get()))

        xriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
        xriL = np.frombuffer(xriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)
        yriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
        yriL = np.frombuffer(yriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)
        uriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
        uriL = np.frombuffer(uriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)
        vriL_mem = RawArray(ctypes.c_double, (Images + 1) * (Ny.get() + 1) * (Nx.get() + 1))
        vriL = np.frombuffer(vriL_mem, dtype=np.float64).reshape(Images + 1, Ny.get() + 1, Nx.get() + 1)

        for i in range(0, Ny.get()+1):
            for j in range(0, Nx.get()+1):
                xriL[0,i,j] = xinitline*Opi.get() + ((j)*dxline*Opi.get())/(Nx.get())
                yriL[0,i,j] = yinitline*Opi.get() + ((i)*dyline*Opi.get())/(Ny.get())

        if NumCut.get() != 0:
            if 'Rectangular' in TypeCut.get():
                for i in range(0,NumCut.get()):
                    console.insert(tk.END, f'Select the rectangular cut region # {str(i+1)} and press enter key\n\n')
                    console.see('insert')
                    image = cv.imread(fileNamesSelection[0])
                    for i in range(0, Ny.get()):
                        for j in range(0, Nx.get()):
                            if np.isnan(xriL[0][i][int((Nx.get()+1)/2)]) or np.isnan(xriL[0][i+1][int((Nx.get()+1)/2)]):
                                pass
                            else:
                                image = cv.line(image,(int(xriL[0][i][int((Nx.get()+1)/2)]/Opi.get()),int(yriL[0][i][int((Nx.get()+1)/2)]/Opi.get())),(int(xriL[0][i+1][int((Nx.get()+1)/2)]/Opi.get()),int(yriL[0][i+1][int((Nx.get()+1)/2)]/Opi.get())),(255, 0, 0),3)
                            if np.isnan(xriL[0][int((Ny.get()+1)/2)][j]) or np.isnan(xriL[0][int((Ny.get()+1)/2)][j+1]):
                                pass
                            else:
                                image = cv.line(image,(int(xriL[0][int((Ny.get()+1)/2)][j]/Opi.get()),int(yriL[0][int((Ny.get()+1)/2)][j]/Opi.get())),(int(xriL[0][int((Ny.get()+1)/2)][j+1]/Opi.get()),int(yriL[0][int((Ny.get()+1)/2)][j+1]/Opi.get())),(255, 0, 0),3)

                    for i in range(0, Ny.get()+1):
                        for j in range(0, Nx.get()):
                            if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i][j+1]):
                                pass
                            else:
                                image = cv.line(image,(int(xriL[0][i][j]/Opi.get()),int(yriL[0][i][j]/Opi.get())),(int(xriL[0][i][j+1]/Opi.get()),int(yriL[0][i][j+1]/Opi.get())),(0, 0, 255),2)

                    for i in range(0, Ny.get()):
                        for j in range(0, Nx.get()+1):
                            if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i+1][j]):
                                pass
                            else:
                                image = cv.line(image,(int(xriL[0][i][j]/Opi.get()),int(yriL[0][i][j]/Opi.get())),(int(xriL[0][i+1][j]/Opi.get()),int(yriL[0][i+1][j]/Opi.get())),(0, 0, 255),2)

                    dxfigure = int(image.shape[1])
                    dyfigure = int(image.shape[0])

                    screen_width = menu.winfo_screenwidth() - 100
                    screen_height = menu.winfo_screenheight() - 100

                    # Change size of displayed image:
                    if dyfigure > screen_height:
                        ratio = screen_height / dyfigure
                        image_figure = cv.resize(image,
                                                 (int(screen_height * dxfigure / dyfigure), screen_height))
                        #image_figure = cv.cvtColor(image_figure, cv.COLOR_GRAY2RGB)
                        xinitline, yinitline, dxline, dyline = SelectROI_DIC(menu, console,'Select the rectangular cut region',image_figure)

                        xinitline = xinitline / ratio
                        yinitline = yinitline / ratio
                        dxline = dxline / ratio
                        dyline = dyline / ratio
                    else:
                        xinitline, yinitline, dxline, dyline = SelectROI_DIC(menu, console,'Select the rectangular cut region',image)

                    cv.destroyAllWindows()

                    for i in range(0, Ny.get()+1):
                        for j in range(0, Nx.get()+1):

                            if xinitline <= xriL[0][i][j]/Opi.get() <= xinitline+dxline and yinitline <= yriL[0][i][j]/Opi.get() <= yinitline+dyline:
                                xriL[0][i][j] = float('nan')
                                yriL[0][i][j] = float('nan')

            if 'Circular inside' in TypeCut.get():
                for i in range(0, NumCut.get()):
                    console.insert(tk.END,
                                   f'Select the circular cutting region # {str(i + 1)} and press enter key\n\n')
                    console.see('insert')

                    image = cv.imread(fileNamesSelection[0])
                    for i in range(0, Ny.get()):
                        for j in range(0, Nx.get()):
                            if np.isnan(xriL[0][i][int((Nx.get() + 1) / 2)]) or np.isnan(
                                    xriL[0][i + 1][int((Nx.get() + 1) / 2)]):
                                pass
                            else:
                                image = cv.line(image, (int(xriL[0][i][int((Nx.get() + 1) / 2)] / Opi.get()),
                                                        int(yriL[0][i][int((Nx.get() + 1) / 2)] / Opi.get())), (
                                                    int(xriL[0][i + 1][int((Nx.get() + 1) / 2)] / Opi.get()),
                                                    int(yriL[0][i + 1][int((Nx.get() + 1) / 2)] / Opi.get())),
                                                (255, 0, 0), 3)
                            if np.isnan(xriL[0][int((Ny.get() + 1) / 2)][j]) or np.isnan(
                                    xriL[0][int((Ny.get() + 1) / 2)][j + 1]):
                                pass
                            else:
                                image = cv.line(image, (int(xriL[0][int((Ny.get() + 1) / 2)][j] / Opi.get()),
                                                        int(yriL[0][int((Ny.get() + 1) / 2)][j] / Opi.get())), (
                                                    int(xriL[0][int((Ny.get() + 1) / 2)][j + 1] / Opi.get()),
                                                    int(yriL[0][int((Ny.get() + 1) / 2)][j + 1] / Opi.get())),
                                                (255, 0, 0), 3)

                    for i in range(0, Ny.get() + 1):
                        for j in range(0, Nx.get()):
                            if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i][j + 1]):
                                pass
                            else:
                                image = cv.line(image, (int(xriL[0][i][j] / Opi.get()), int(yriL[0][i][j] / Opi.get())),
                                                (
                                                int(xriL[0][i][j + 1] / Opi.get()), int(yriL[0][i][j + 1] / Opi.get())),
                                                (0, 0, 255), 2)

                    for i in range(0, Ny.get()):
                        for j in range(0, Nx.get() + 1):
                            if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i + 1][j]):
                                pass
                            else:
                                image = cv.line(image, (int(xriL[0][i][j] / Opi.get()), int(yriL[0][i][j] / Opi.get())),
                                                (
                                                int(xriL[0][i + 1][j] / Opi.get()), int(yriL[0][i + 1][j] / Opi.get())),
                                                (0, 0, 255), 2)

                    dxfigure = int(image.shape[1])
                    dyfigure = int(image.shape[0])

                    screen_width = menu.winfo_screenwidth() - 100
                    screen_height = menu.winfo_screenheight() - 100

                    # Change size of displayed image:
                    if dyfigure > screen_height:
                        ratio = screen_height / dyfigure
                        image_figure = cv.resize(image,
                                                 (int(screen_height * dxfigure / dyfigure), screen_height))
                        # image_figure = cv.cvtColor(image_figure, cv.COLOR_GRAY2RGB)
                        pointsCut = CircularROI_DIC(menu, console, 'Select the circular cut region', image_figure)

                        for w in range(0, len(pointsCut)):
                            pointsCut[w] = [x / ratio for x in pointsCut[w]]

                    else:
                        pointsCut = CircularROI_DIC(menu, console, 'Select the circular cut region', image)

                    cv.destroyAllWindows()

                    polygon = Polygon(pointsCut)
                    for i in range(0, Ny.get() + 1):
                        for j in range(0, Nx.get() + 1):
                            point = Point(xriL[0][i][j] / Opi.get(), yriL[0][i][j] / Opi.get())
                            if polygon.contains(point):
                                xriL[0][i][j] = float('nan')
                                yriL[0][i][j] = float('nan')
                                uriL[0][i][j] = float('nan')
                                vriL[0][i][j] = float('nan')

            if 'Circular outside' in TypeCut.get():
                for i in range(0, NumCut.get()):
                    console.insert(tk.END,
                                   f'Select the circular cutting region # {str(i + 1)} and press enter key\n\n')
                    console.see('insert')

                    image = cv.imread(fileNamesSelection[0])
                    for i in range(0, Ny.get()):
                        for j in range(0, Nx.get()):
                            if np.isnan(xriL[0][i][int((Nx.get() + 1) / 2)]) or np.isnan(
                                    xriL[0][i + 1][int((Nx.get() + 1) / 2)]):
                                pass
                            else:
                                image = cv.line(image, (int(xriL[0][i][int((Nx.get() + 1) / 2)] / Opi.get()),
                                                        int(yriL[0][i][int((Nx.get() + 1) / 2)] / Opi.get())), (
                                                    int(xriL[0][i + 1][int((Nx.get() + 1) / 2)] / Opi.get()),
                                                    int(yriL[0][i + 1][int((Nx.get() + 1) / 2)] / Opi.get())),
                                                (255, 0, 0), 3)
                            if np.isnan(xriL[0][int((Ny.get() + 1) / 2)][j]) or np.isnan(
                                    xriL[0][int((Ny.get() + 1) / 2)][j + 1]):
                                pass
                            else:
                                image = cv.line(image, (int(xriL[0][int((Ny.get() + 1) / 2)][j] / Opi.get()),
                                                        int(yriL[0][int((Ny.get() + 1) / 2)][j] / Opi.get())), (
                                                    int(xriL[0][int((Ny.get() + 1) / 2)][j + 1] / Opi.get()),
                                                    int(yriL[0][int((Ny.get() + 1) / 2)][j + 1] / Opi.get())),
                                                (255, 0, 0), 3)

                    for i in range(0, Ny.get() + 1):
                        for j in range(0, Nx.get()):
                            if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i][j + 1]):
                                pass
                            else:
                                image = cv.line(image, (int(xriL[0][i][j] / Opi.get()), int(yriL[0][i][j] / Opi.get())),
                                                (
                                                int(xriL[0][i][j + 1] / Opi.get()), int(yriL[0][i][j + 1] / Opi.get())),
                                                (0, 0, 255), 2)

                    for i in range(0, Ny.get()):
                        for j in range(0, Nx.get() + 1):
                            if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i + 1][j]):
                                pass
                            else:
                                image = cv.line(image, (int(xriL[0][i][j] / Opi.get()), int(yriL[0][i][j] / Opi.get())),
                                                (
                                                int(xriL[0][i + 1][j] / Opi.get()), int(yriL[0][i + 1][j] / Opi.get())),
                                                (0, 0, 255), 2)

                    dxfigure = int(image.shape[1])
                    dyfigure = int(image.shape[0])

                    screen_width = menu.winfo_screenwidth() - 100
                    screen_height = menu.winfo_screenheight() - 100

                    # Change size of displayed image:
                    if dyfigure > screen_height:
                        ratio = screen_height / dyfigure
                        image_figure = cv.resize(image,
                                                 (int(screen_height * dxfigure / dyfigure), screen_height))
                        # image_figure = cv.cvtColor(image_figure, cv.COLOR_GRAY2RGB)
                        pointsCut = CircularROI_DIC(menu, console, 'Select the circular cut region', image_figure)

                        for w in range(0, len(pointsCut)):
                            pointsCut[w] = [x / ratio for x in pointsCut[w]]

                    else:
                        pointsCut = CircularROI_DIC(menu, console, 'Select the circular cut region', image)

                    cv.destroyAllWindows()

                    polygon = Polygon(pointsCut)
                    for i in range(0, Ny.get() + 1):
                        for j in range(0, Nx.get() + 1):
                            point = Point(xriL[0][i][j] / Opi.get(), yriL[0][i][j] / Opi.get())
                            if polygon.contains(point):
                                pass
                            else:
                                xriL[0][i][j] = float('nan')
                                yriL[0][i][j] = float('nan')
                                uriL[0][i][j] = float('nan')
                                vriL[0][i][j] = float('nan')

            if 'Free' in TypeCut.get():
                for i in range(0,NumCut.get()):
                    console.insert(tk.END, f'Select the freehand cutting region # {str(i+1)} and press enter key\n\n')
                    console.see('insert')

                    image = cv.imread(fileNamesSelection[0])
                    for i in range(0, Ny.get()):
                        for j in range(0, Nx.get()):
                            if np.isnan(xriL[0][i][int((Nx.get()+1)/2)]) or np.isnan(xriL[0][i+1][int((Nx.get()+1)/2)]):
                                pass
                            else:
                                image = cv.line(image,(int(xriL[0][i][int((Nx.get()+1)/2)]/Opi.get()),int(yriL[0][i][int((Nx.get()+1)/2)]/Opi.get())),(int(xriL[0][i+1][int((Nx.get()+1)/2)]/Opi.get()),int(yriL[0][i+1][int((Nx.get()+1)/2)]/Opi.get())),(255, 0, 0),3)
                            if np.isnan(xriL[0][int((Ny.get()+1)/2)][j]) or np.isnan(xriL[0][int((Ny.get()+1)/2)][j+1]):
                                pass
                            else:
                                image = cv.line(image,(int(xriL[0][int((Ny.get()+1)/2)][j]/Opi.get()),int(yriL[0][int((Ny.get()+1)/2)][j]/Opi.get())),(int(xriL[0][int((Ny.get()+1)/2)][j+1]/Opi.get()),int(yriL[0][int((Ny.get()+1)/2)][j+1]/Opi.get())),(255, 0, 0),3)

                    for i in range(0, Ny.get()+1):
                        for j in range(0, Nx.get()):
                            if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i][j+1]):
                                pass
                            else:
                                image = cv.line(image,(int(xriL[0][i][j]/Opi.get()),int(yriL[0][i][j]/Opi.get())),(int(xriL[0][i][j+1]/Opi.get()),int(yriL[0][i][j+1]/Opi.get())),(0, 0, 255),2)

                    for i in range(0, Ny.get()):
                        for j in range(0, Nx.get()+1):
                            if np.isnan(xriL[0][i][j]) or np.isnan(xriL[0][i+1][j]):
                                pass
                            else:
                                image = cv.line(image,(int(xriL[0][i][j]/Opi.get()),int(yriL[0][i][j]/Opi.get())),(int(xriL[0][i+1][j]/Opi.get()),int(yriL[0][i+1][j]/Opi.get())),(0, 0, 255),2)

                    dxfigure = int(image.shape[1])
                    dyfigure = int(image.shape[0])

                    screen_width = menu.winfo_screenwidth() - 100
                    screen_height = menu.winfo_screenheight() - 100

                    # Change size of displayed image:
                    if dyfigure > screen_height:
                        ratio = screen_height / dyfigure
                        image_figure = cv.resize(image,
                                                 (int(screen_height * dxfigure / dyfigure), screen_height))
                        #image_figure = cv.cvtColor(image_figure, cv.COLOR_GRAY2RGB)
                        pointsCut = freehandCut('Select the freehand cutting region', image_figure)

                        for w in range(0,len(pointsCut)):
                            pointsCut[w] = [x/ratio for x in pointsCut[w]]

                    else:
                        pointsCut = freehandCut('Select the freehand cutting region', image)

                    cv.destroyAllWindows()

                    polygon = Polygon(pointsCut)
                    for i in range(0, Ny.get()+1):
                        for j in range(0, Nx.get()+1):
                            point = Point(xriL[0][i][j]/Opi.get(),yriL[0][i][j]/Opi.get())
                            if polygon.contains(point):
                                xriL[0][i][j] = float('nan')
                                yriL[0][i][j] = float('nan')
                                uriL[0][i][j] = float('nan')
                                vriL[0][i][j] = float('nan')

        fig = plt.figure()
        ax = fig.gca()
        dxplot = int(cv.imread(fileNamesSelection[0]).shape[1])
        dyplot = int(cv.imread(fileNamesSelection[0]).shape[0])

        ratio_plot = dxplot / dyplot;

        if ratio_plot <= 1.33333333:

            dxplotdark = dyplot * 1.33333333
            dyplotdark = dyplot
            ax.imshow(cv.imread(fileNamesSelection[0]), zorder=2)
            ax.plot(np.transpose(xriL[0][:][:] / Opi.get()), np.transpose(yriL[0][:][:] / Opi.get()), color='red',
                    linewidth=1, zorder=3)
            ax.plot(xriL[0][:][:] / Opi.get(), yriL[0][:][:] / Opi.get(), color='red', linewidth=1, zorder=3)

            winSub = False
            for i in range(0, Ny.get() + 1):
                for j in range(0, Nx.get() + 1):
                    if ~np.isnan(xriL[0][i][j]):

                        ax.plot(
                            [xriL[0][i][j] / Opi.get() - SubIr.get() / 2, xriL[0][i][j] / Opi.get() + SubIr.get() / 2,
                             xriL[0][i][j] / Opi.get() + SubIr.get() / 2, xriL[0][i][j] / Opi.get() - SubIr.get() / 2,
                             xriL[0][i][j] / Opi.get() - SubIr.get() / 2],
                            [yriL[0][i][j] / Opi.get() + SubIr.get() / 2, yriL[0][i][j] / Opi.get() + SubIr.get() / 2,
                             yriL[0][i][j] / Opi.get() - SubIr.get() / 2, yriL[0][i][j] / Opi.get() - SubIr.get() / 2,
                             yriL[0][i][j] / Opi.get() + SubIr.get() / 2],
                            color='#00FF00', linewidth=1, zorder=3)  # SubIr window

                        ax.plot(
                            [xriL[0][i][j] / Opi.get() - SubIb.get() / 2, xriL[0][i][j] / Opi.get() + SubIb.get() / 2,
                             xriL[0][i][j] / Opi.get() + SubIb.get() / 2, xriL[0][i][j] / Opi.get() - SubIb.get() / 2,
                             xriL[0][i][j] / Opi.get() - SubIb.get() / 2],
                            [yriL[0][i][j] / Opi.get() + SubIb.get() / 2, yriL[0][i][j] / Opi.get() + SubIb.get() / 2,
                             yriL[0][i][j] / Opi.get() - SubIb.get() / 2, yriL[0][i][j] / Opi.get() - SubIb.get() / 2,
                             yriL[0][i][j] / Opi.get() + SubIb.get() / 2],
                            color='blue', linewidth=1, zorder=3)  # SubIb window

                        winSub = True
                        break
                    else:
                        continue

                if winSub: break

            ax.plot([0 - dxplotdark / 2 + dxplot / 2, 0 - dxplotdark / 2 + dxplot / 2,
                     dxplotdark - dxplotdark / 2 + dxplot / 2, dxplotdark - dxplotdark / 2 + dxplot / 2,
                     0 - dxplotdark / 2 + dxplot / 2], [0, dyplotdark, dyplotdark, 0, 0], 'black', zorder=1)
            ax.fill_between([0 - dxplotdark / 2 + dxplot / 2, 0 - dxplotdark / 2 + dxplot / 2,
                             dxplotdark - dxplotdark / 2 + dxplot / 2, dxplotdark - dxplotdark / 2 + dxplot / 2,
                             0 - dxplotdark / 2 + dxplot / 2], [0, dyplotdark, dyplotdark, 0, 0], color='black',
                            zorder=1)
            ax.axis('off')
            plt.xlim(-dxplotdark / 2 + dxplot / 2, dxplotdark - dxplotdark / 2 + dxplot / 2)
            plt.ylim(0, dyplotdark)
            plt.subplots_adjust(0, 0, 1, 1, 0, 0)

        else:
            dxplotdark = dxplot
            dyplotdark = dxplot / 1.33333333
            ax.imshow(cv.imread(fileNamesSelection[0]), zorder=2)
            ax.plot(np.transpose(xriL[0][:][:] / Opi.get()), np.transpose(yriL[0][:][:] / Opi.get()), color='red',
                    linewidth=1, zorder=3)
            ax.plot(xriL[0][:][:] / Opi.get(), yriL[0][:][:] / Opi.get(), color='red', linewidth=1, zorder=3)

            winSub = False
            for i in range(0, Ny.get() + 1):
                 for j in range(0, Nx.get() + 1):
                    if ~np.isnan(xriL[0][i][j]):

                        ax.plot(
                            [xriL[0][i][j] / Opi.get() - SubIr.get() / 2, xriL[0][i][j] / Opi.get() + SubIr.get() / 2,
                             xriL[0][i][j] / Opi.get() + SubIr.get() / 2, xriL[0][i][j] / Opi.get() - SubIr.get() / 2,
                             xriL[0][i][j] / Opi.get() - SubIr.get() / 2],
                            [yriL[0][i][j] / Opi.get() + SubIr.get() / 2, yriL[0][i][j] / Opi.get() + SubIr.get() / 2,
                             yriL[0][i][j] / Opi.get() - SubIr.get() / 2, yriL[0][i][j] / Opi.get() - SubIr.get() / 2,
                             yriL[0][i][j] / Opi.get() + SubIr.get() / 2],
                            color='#00FF00', linewidth=1, zorder=3)  # SubIr window

                        ax.plot(
                            [xriL[0][i][j] / Opi.get() - SubIb.get() / 2, xriL[0][i][j] / Opi.get() + SubIb.get() / 2,
                             xriL[0][i][j] / Opi.get() + SubIb.get() / 2, xriL[0][i][j] / Opi.get() - SubIb.get() / 2,
                             xriL[0][i][j] / Opi.get() - SubIb.get() / 2],
                            [yriL[0][i][j] / Opi.get() + SubIb.get() / 2, yriL[0][i][j] / Opi.get() + SubIb.get() / 2,
                             yriL[0][i][j] / Opi.get() - SubIb.get() / 2, yriL[0][i][j] / Opi.get() - SubIb.get() / 2,
                             yriL[0][i][j] / Opi.get() + SubIb.get() / 2],
                            color='blue', linewidth=1, zorder=3)  # SubIb window

                        winSub = True
                        break
                    else:
                        continue

                 if winSub: break

            ax.plot([0, 0, dxplotdark, dxplotdark, 0],
                    [0 - dyplotdark / 2 + dyplot / 2, +dyplotdark - dyplotdark / 2 + dyplot / 2,
                     +dyplotdark - dyplotdark / 2 + dyplot / 2, 0 - dyplotdark / 2 + dyplot / 2,
                     0 - dyplotdark / 2 + dyplot / 2], 'black', zorder=1)
            ax.fill_between([0, 0, dxplotdark, dxplotdark, 0],
                            [0 - dyplotdark / 2 + dyplot / 2, +dyplotdark - dyplotdark / 2 + dyplot / 2,
                             +dyplotdark - dyplotdark / 2 + dyplot / 2, 0 - dyplotdark / 2 + dyplot / 2,
                             0 - dyplotdark / 2 + dyplot / 2], color='black', zorder=1)
            ax.axis('off')
            plt.xlim(0, dxplotdark)
            plt.ylim(-dyplotdark / 2 + dyplot / 2, dyplotdark - dyplotdark / 2 + dyplot / 2)
            plt.subplots_adjust(0, 0, 1, 1, 0, 0)

        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        image = np.flip(image.reshape((h, w, 3)), axis=0)

        plt.cla()
        plt.clf()

        image = cv.resize(image, (640, 480))
        cv.putText(image, f'CORRELATION MESH - {np.count_nonzero(~np.isnan(xriL[0][:][:]))} nodes', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
        cv.putText(image, 'REFERENCE SUBSET', (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(image, 'SEARCH SUBSET', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

        image = ImageTk.PhotoImage(Image.fromarray(image))

        canvas.image = image
        canvas.configure(image=image)

        console.insert(tk.END, 'Start the correlation process\n\n')
        console.see('insert')

        messagebox.showinfo('Information','Press start to initialize the correlation!')

########################################################################################################################
# Initialize function
########################################################################################################################
def initialize(menu, console, file_var, V, file, capturedFolder, SubIr, SubIb, Nx,
         Ny, Valpixel, Valmm, Opi, OpiSub, Version, TypeCut, NumCut, Adjust, Alpha, progression, progression_bar, canvas,
               canvas_text, process_btn, abort_param, console_process, Method, Correlation, Criterion, Step, method_corr_dict, Interpolation, Filtering, Kernel, ResultsName, Cores):
    global fileNames, t2

    try:
        fileNames and xriL
    except NameError:
        messagebox.showerror('Error','Please load project or select the image captured folder and DIC settings before starting the correlation process!')
    else:

        if SubIr.get() >= SubIb.get():
            messagebox.showerror('Error','The search subset size must be larger than the reference subset size. Please change the input data!')
        else:
            save(menu, console, file_var, V, file, capturedFolder, SubIr, SubIb, Nx,
         Ny, Valpixel, Valmm, Opi, OpiSub, Version, TypeCut, NumCut, Adjust, Alpha, Method, Correlation, Criterion, Step, Interpolation, Filtering, Kernel)

            process_btn.configure(text='Abort',fg='#cc0000',command = lambda: abort(abort_param))

            t2 = Thread(target=CorrelationProcess, args=(menu, console, file_var, V, file, capturedFolder, SubIr, SubIb, Nx,
         Ny, Valpixel, Valmm, Opi, OpiSub, Version, TypeCut, NumCut, Adjust, Alpha, progression, progression_bar, canvas, canvas_text,
         process_btn, abort_param, console_process, Method, Correlation, Criterion, Step, method_corr_dict, Interpolation, Filtering, Kernel, ResultsName, Cores))
            t2.setDaemon(True)
            t2.start()

########################################################################################################################
# Time function
########################################################################################################################
def second2dhms(sec):
    day = sec // (24 * 3600)
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60
    seconds = sec

    return '%02d - %02d:%02d:%02d' % (day, hour, minutes, seconds)

########################################################################################################################
# 2D DIC main
########################################################################################################################
def CorrelationProcess(menu, console, file_var, V, file, capturedFolder, SubIr, SubIb, Nx,
         Ny, Valpixel, Valmm, Opi, OpiSub, Version, TypeCut, NumCut, Adjust, Alpha, progression, progression_bar, canvas, canvas_text,
                       process_btn, abort_param, console_process, Method, Correlation, Criterion, Step, method_corr_dict, Interpolation, Filtering, Kernel, ResultsName, Cores):
    global fileNamesSelection, selectionPath, resultsPath, fileNames, Format, Images, calib, xriL, yriL, uriL, vriL
    global xriL_mem, yriL_mem, uriL_mem, vriL_mem, xriR_mem, yriR_mem, uriR_mem, vriR_mem

    abort_param.set(False)

    if Interpolation.get() == 'After':
        Opi.set(1)
        console.insert(tk.END,
                       f'The interpolation factor was changed to 1 according to the interpolation preference (after correlation)!\n\n')
        console.see('insert')

    resultsPath = capturedFolder.get().rsplit('/', 1)[0] + f'/{ResultsName.get()}'
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)
        console.insert(tk.END, f'The {ResultsName.get()} folder was created\n\n')
        console.see('insert')
    else:
        console.insert(tk.END, f'The {ResultsName.get()} folder is already in the main directory\n\n')
        console.see('insert')

    for files in os.listdir(resultsPath):
        if files.endswith(".dat"):
            os.remove(os.path.join(resultsPath, files))

    time_iter = np.zeros(Images+1)

    # Correlation functions:
    method_var = ['cv.TM_CCOEFF','cv.TM_CCOEFF_NORMED','cv.TM_CCORR','cv.TM_CCORR_NORMED']
    matchTemplate_method = eval(f'{method_var[method_corr_dict.get(Method.get())]}')

    filex='xm{:02d}.dat'; filey='ym{:02d}.dat'; fileu='um{:02d}.dat'; filev='vm{:02d}.dat'

    # Divide the subset points in groups to perform the parallel computing:
    cores_x = int(Cores.get() / 2 + 1)
    cores_y = int(Cores.get() / (cores_x - 1) + 1)

    ii = np.linspace(0, Ny.get() + 1, num=cores_y, dtype="int")
    i1 = ii[1:]
    i0 = ii[0:cores_y - 1];
    i0[0] = 0

    jj = np.linspace(0, Nx.get() + 1, num=cores_x, dtype="int")
    j1 = jj[1:]
    j0 = jj[0:cores_x - 1];
    j0[0] = 0

    i0_list = np.repeat(i0, cores_x - 1)
    i1_list = np.repeat(i1, cores_x - 1)

    j0_list = np.tile(j0, cores_y - 1)
    j1_list = np.tile(j1, cores_y - 1)

    console.insert(tk.END, 'The process has started\n\n')

    start_process = time.time()

    start = time.time()

    progression.coords(progression_bar, 0, 0, 0, 25); progression.itemconfig(canvas_text, text='')

    console.insert(tk.END, '################################ Correlation parameters ###################################\n\n')
    console.insert(tk.END, f'Interpolation type ............................................... {Interpolation.get()}\n')
    console.insert(tk.END, f'Image interpolation factor ....................................... {Opi.get()}\n')
    console.insert(tk.END, f'Subpixel after correlation ....................................... {OpiSub.get()}\n')
    console.insert(tk.END, f'Contrast adjust .................................................. {Adjust.get()}\n')
    console.insert(tk.END, f'Search subset size (SSS) ......................................... {SubIb.get()}\n')
    console.insert(tk.END, f'Reference subset size (RSS) ...................................... {SubIr.get()}\n')
    console.insert(tk.END, f'Subset step (ST) ................................................. {Step.get()}\n')
    console.insert(tk.END, f'Number of steps in x direction ................................... {Nx.get()}\n')
    console.insert(tk.END, f'Number of steps in y direction ................................... {Ny.get()}\n')
    console.insert(tk.END, f'Value in pixels for calibration .................................. {Valpixel.get()}\n')
    console.insert(tk.END, f'Value in mm for calibration ...................................... {Valmm.get()}\n')
    console.insert(tk.END, f'Selected description ............................................. {Version.get()}\n')
    console.insert(tk.END, f'Correlation type ................................................. {Correlation.get()}\n')
    console.insert(tk.END, f'Correlation function ............................................. {method_var[method_corr_dict.get(Method.get())]}\n')
    console.insert(tk.END, f'Correlation criterion ............................................ {Criterion.get()}\n')
    console.insert(tk.END, f'Filtering ........................................................ {Filtering.get()}\n')
    console.insert(tk.END, f'Kernel size for filtering ........................................ {Kernel.get()}\n')
    console.insert(tk.END, f'Calculation points ............................................... {np.count_nonzero(~np.isnan(xriL[0][:][:]))}\n')
    console.insert(tk.END, f'Number of processors ............................................. {Cores.get()}\n\n')
    console.insert(tk.END, '###########################################################################################\n\n')
    console.see('insert')

    console.insert(tk.END, 'Creation of variables ')
    console.see('insert')

    # Reference stereo pair:
    I0 = cv.cvtColor(cv.imread(fileNamesSelection[0]), cv.COLOR_BGR2GRAY)

    Iref = cv.resize(I0, (int(I0.shape[1] * Opi.get()), int(I0.shape[0] * Opi.get())), interpolation=cv.INTER_CUBIC)

    Cal = Valmm.get()/Valpixel.get()

    xm = xriL[0][:][:] * Cal / Opi.get()
    ym = yriL[0][:][:] * Cal / Opi.get()
    um = uriL[0][:][:] * Cal / Opi.get()
    vm = vriL[0][:][:] * Cal / Opi.get()

    np.savetxt(f'{resultsPath}/{filex.format(1)}', xm, fmt='%.10e')
    np.savetxt(f'{resultsPath}/{filey.format(1)}', ym, fmt='%.10e')
    np.savetxt(f'{resultsPath}/{fileu.format(1)}', um,  fmt='%.10e')
    np.savetxt(f'{resultsPath}/{filev.format(1)}', vm,  fmt='%.10e')

    SubIrV = SubIr.get() * Opi.get()
    SubIbV = SubIb.get() * Opi.get()

    console.insert(tk.END, '- Done\n')
    console.see('insert')

    time_total = []

    start_process = time.time()

    for k in range(1, Images):

        start = time.time()

        if abort_param.get():
            process_btn.configure(text='Start',fg ='#282C34', command= lambda: initialize(menu, console, file_var, V, file, capturedFolder, SubIr, SubIb, Nx,
     Ny, Valpixel, Valmm, Opi, OpiSub, Version, TypeCut, NumCut, Adjust, Alpha, progression, progression_bar, canvas, canvas_text, process_btn, abort_param, console_process,
                                                                                          Method, Correlation, Criterion, Step, method_corr_dict, Interpolation, Filtering, Kernel,ResultsName, Cores))
            console.insert(tk.END, '\nCorrelation process was aborted . . .\n\n')
            console.see('insert')
            abort_param.set(False)
            return
        else:

            console.insert(tk.END, f'Image correlation - Process {k} of {Images} ')
            console.see('insert')

            I0 = cv.cvtColor(cv.imread(fileNamesSelection[k - 1]), cv.COLOR_BGR2GRAY)
            I1 = cv.cvtColor(cv.imread(fileNamesSelection[k]), cv.COLOR_BGR2GRAY)

            Iun = cv.resize(I0, (int(I0.shape[1] * Opi.get()), int(I0.shape[0] * Opi.get())),
                             interpolation=cv.INTER_CUBIC)
            Id = cv.resize(I1, (int(I1.shape[1] * Opi.get()), int(I1.shape[0] * Opi.get())),
                            interpolation=cv.INTER_CUBIC)

            if Interpolation.get() == 'Before':
                if Correlation.get() == 'Incremental':
                    process_list = []
                    for i in range(Cores.get()):
                        process_list.append(mp.Process(target=Corr2D_V1, args=(k-1, k, SubIrV, SubIbV,
                                                                                      xriL_mem, xriL.shape,
                                                                                      yriL_mem, yriL.shape,
                                                                                      uriL_mem, uriL.shape,
                                                                                      vriL_mem, vriL.shape,
                                                                                      Iun, Id,
                                                                                      Version.get(),
                                                                                      matchTemplate_method,
                                                                                      Criterion.get(), i0_list[i],
                                                                                      i1_list[i], j0_list[i],
                                                                                      j1_list[i])))
                        try:
                            process_list[i].start()
                        except:
                            messagebox.showerror('Error RAM',
                                                 'The correlation runs out of memory RAM! Please reduce the number of designated processors or reduce the kb interpolation factor.')

                            menu.destroy()
                            menu.quit()

                    for i in range(Cores.get()):
                        process_list[i].join()

                else:
                    process_list = []
                    for i in range(Cores.get()):
                        process_list.append(mp.Process(target=Corr2D_V1, args=(0, k, SubIrV, SubIbV,
                                                                          xriL_mem, xriL.shape,
                                                                          yriL_mem, yriL.shape,
                                                                          uriL_mem, uriL.shape,
                                                                          vriL_mem, vriL.shape,
                                                                          Iref, Id,
                                                                          Version.get(),
                                                                          matchTemplate_method,
                                                                          Criterion.get(), i0_list[i],
                                                                          i1_list[i], j0_list[i],
                                                                          j1_list[i])))

                        try:
                            process_list[i].start()
                        except:
                            messagebox.showerror('Error RAM',
                                                 'The correlation runs out of memory RAM! Please reduce the number of designated processors or reduce the kb interpolation factor.')

                            menu.destroy()
                            menu.quit()

                    for i in range(Cores.get()):
                        process_list[i].join()
            else:
                if Correlation.get() == 'Incremental':
                    process_list = []
                    for i in range(Cores.get()):
                        process_list.append(mp.Process(target=Corr2D_V2, args=(k-1, k, SubIrV, SubIbV, OpiSub.get(),
                                                                          xriL_mem, xriL.shape,
                                                                          yriL_mem, yriL.shape,
                                                                          uriL_mem, uriL.shape,
                                                                          vriL_mem, vriL.shape,
                                                                          Iun, Id,
                                                                          Version.get(),
                                                                          matchTemplate_method,
                                                                          Criterion.get(), i0_list[i],
                                                                          i1_list[i], j0_list[i],
                                                                          j1_list[i])))
                        try:
                            process_list[i].start()
                        except:
                            messagebox.showerror('Error RAM',
                                                 'The correlation runs out of memory RAM! Please reduce the number of designated processors or reduce the kb interpolation factor.')

                            menu.destroy()
                            menu.quit()

                    for i in range(Cores.get()):
                        process_list[i].join()

                else:
                    process_list = []
                    for i in range(Cores.get()):
                        process_list.append(mp.Process(target=Corr2D_V2, args=(0, k, SubIrV, SubIbV,  OpiSub.get(),
                                                                          xriL_mem, xriL.shape,
                                                                          yriL_mem, yriL.shape,
                                                                          uriL_mem, uriL.shape,
                                                                          vriL_mem, vriL.shape,
                                                                          Iref, Id,
                                                                          Version.get(),
                                                                          matchTemplate_method,
                                                                          Criterion.get(), i0_list[i],
                                                                          i1_list[i], j0_list[i],
                                                                          j1_list[i])))

                        try:
                            process_list[i].start()
                        except:
                            messagebox.showerror('Error RAM',
                                                 'The correlation runs out of memory RAM! Please reduce the number of designated processors or reduce the kb interpolation factor.')

                            menu.destroy()
                            menu.quit()

                    for i in range(Cores.get()):
                        process_list[i].join()

            console.insert(tk.END, '- Done\n')
            console.insert(tk.END, 'Saving files ')
            console.see('insert')

            xmL = xriL[k][:][:].copy() * Cal / Opi.get()
            ymL = yriL[k][:][:].copy() * Cal / Opi.get()
            umL = uriL[k][:][:].copy() * Cal / Opi.get()
            vmL = vriL[k][:][:].copy() * Cal / Opi.get()

            # Displacement filtering:
            if Filtering.get() == 'Gaussian':

                # Gaussian smoothing:
                um=cv.GaussianBlur(umL,(Kernel.get(), Kernel.get()), cv.BORDER_DEFAULT)
                vm=cv.GaussianBlur(vmL,(Kernel.get(), Kernel.get()), cv.BORDER_DEFAULT)

                # Determination of valid mask area:
                valid_mask = np.isnan(um)

                # Add nan values according to displacement data:
                for i in range(0, Ny.get() + 1):
                    for j in range(0, Nx.get() + 1):
                        if valid_mask[i][j]:
                            xmL[i][j] = 'nan'
                            ymL[i][j] = 'nan'


                # Position:
                xm = xmL
                ym = ymL

            else:

                xm = xmL
                ym = ymL
                um = umL
                vm = vmL

            np.savetxt(f'{resultsPath}/{filex.format(k+1)}', xm, fmt='%.10e')
            np.savetxt(f'{resultsPath}/{filey.format(k+1)}', ym, fmt='%.10e')
            np.savetxt(f'{resultsPath}/{fileu.format(k+1)}', um,  fmt='%.10e')
            np.savetxt(f'{resultsPath}/{filev.format(k+1)}', vm,  fmt='%.10e')

            fig = plt.figure()
            ax = fig.gca()
            dxplot = int(cv.imread(fileNamesSelection[0]).shape[1])
            dyplot = int(cv.imread(fileNamesSelection[0]).shape[0])

            ratio_plot = dxplot/dyplot;

            if ratio_plot <= 1.33333333:

                dxplotdark = dyplot*1.33333333
                dyplotdark = dyplot
                ax.imshow(cv.imread(fileNamesSelection[k]), zorder=2)
                ax.plot(np.transpose(xriL[k][:][:]/Opi.get()), np.transpose(yriL[k][:][:]/Opi.get()), color= 'red', linewidth=1, zorder=3)
                ax.plot(xriL[k][:][:]/Opi.get(), yriL[k][:][:]/Opi.get(), color= 'red', linewidth=1, zorder=3)
                ax.plot([0-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2, dxplotdark-dxplotdark/2+dxplot/2, dxplotdark-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2],[0, dyplotdark, dyplotdark, 0, 0],'black', zorder=1)
                ax.fill_between([0-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2, dxplotdark-dxplotdark/2+dxplot/2, dxplotdark-dxplotdark/2+dxplot/2, 0-dxplotdark/2+dxplot/2],[0, dyplotdark, dyplotdark, 0, 0], color='black', zorder=1)
                ax.axis('off')
                plt.xlim(-dxplotdark/2+dxplot/2,dxplotdark-dxplotdark/2+dxplot/2)
                plt.ylim(0,dyplotdark)
                plt.subplots_adjust(0,0,1,1,0,0)

            else:
                dxplotdark = dxplot
                dyplotdark = dxplot/1.33333333
                ax.imshow(cv.imread(fileNamesSelection[k]), zorder=2)
                ax.plot(np.transpose(xriL[k][:][:]/Opi.get()), np.transpose(yriL[k][:][:]/Opi.get()), color= 'red', linewidth=1, zorder=3)
                ax.plot(xriL[k][:][:]/Opi.get(), yriL[k][:][:]/Opi.get(), color= 'red', linewidth=1, zorder=3)
                ax.plot([0, 0, dxplotdark, dxplotdark, 0],[0-dyplotdark/2+dyplot/2, +dyplotdark-dyplotdark/2+dyplot/2, +dyplotdark-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2],'black', zorder=1)
                ax.fill_between([0, 0, dxplotdark, dxplotdark, 0],[0-dyplotdark/2+dyplot/2, +dyplotdark-dyplotdark/2+dyplot/2, +dyplotdark-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2, 0-dyplotdark/2+dyplot/2], color='black', zorder=1)
                ax.axis('off')
                plt.xlim(0,dxplotdark)
                plt.ylim(-dyplotdark/2+dyplot/2,dyplotdark-dyplotdark/2+dyplot/2)
                plt.subplots_adjust(0,0,1,1,0,0)

            fig.canvas.draw()

            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            image = np.flip(image.reshape((h, w, 3)),axis=0)

            plt.cla()
            plt.clf()

            image = cv.resize(image, (640, 480))
            cv.putText(image,f'IMAGE - {k+1}',(20,30),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv.LINE_AA)
            image = ImageTk.PhotoImage (Image.fromarray (image))

            canvas.image = image
            canvas.configure(image = image)

            green_length = int(756*((k+1)/Images))
            progression.coords(progression_bar, 0, 0, green_length, 25); progression.itemconfig(canvas_text, text=f'{k+1} of {Images} - {100*(k+1)/Images:.2f}%')

            plt.close('all')

            console.insert(tk.END, '- Done\n\n')

            console.insert(tk.END, '###########################################################################################\n\n')

            end = time.time()
            time_iter[k] = end-start
            time_total = time_total + time_iter[k]

            time_current = second2dhms(time_iter[k])
            time_remaining = second2dhms((Images - k - 1) * time_iter[1])
            time_start = time.strftime('%m/%d/%y - %H:%M:%S', time.localtime(start_process))
            time_end = time.strftime('%m/%d/%y - %H:%M:%S', time.localtime(start_process + (Images - 1) * time_iter[1]))

            console_process.delete('1.0', END)
            console_process.insert(tk.END, f'      Process {k+1} of {Images} \n\n')

            console_process.insert(tk.END, f'Start .. {time_start}\n')
            console_process.insert(tk.END, f'Current ...... {time_current}\n')
            console_process.insert(tk.END, f'Remaining .... {time_remaining}\n\n')

            console_process.insert(tk.END, f'End .... {time_end}\n')
            console.see('insert')

    console.insert(tk.END, 'Correlation - Finished\n')
    console.insert(tk.END, f'The correlation process was successfully completed in {str(datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S"))}\n\n')
    console.see('insert')

    process_btn.configure(text='Start',fg ='#282C34', command= lambda: initialize(menu, console, file_var, V, file, capturedFolder, SubIr, SubIb, Nx,
     Ny, Valpixel, Valmm, Opi, OpiSub, Version, TypeCut, NumCut, Adjust, Alpha, progression, progression_bar, canvas, canvas_text, process_btn, abort_param, console_process,
                                                                                  Method, Correlation, Criterion, Step, method_corr_dict, Interpolation, Filtering, Kernel,ResultsName, Cores))

    abort_param.set(False)

    messagebox.showinfo('Finished','The correlation process has been successfully completed! Use the iCorrVision Post-processing Module for visualization!')
