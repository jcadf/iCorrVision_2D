########################################################################################################################
# iCorrVision-2D Grabber Module                                                                                        #
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
    iCorrVision-2D Grabber Module
    Copyright (C) 2022 iCorrVision team

    This file is part of the iCorrVision-2D software.

    The iCorrVision-2D Grabber Module is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    The iCorrVision-2D Grabber Module is distributed in the hope that it will be useful,
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
import tkinter as tk
from tkinter import *
import os
from tkinter import filedialog
from pypylon import pylon
from threading import Thread
import cv2 as cv
from PIL import Image, ImageTk
import time
from tkinter import messagebox
import subprocess

########################################################################################################################
# Open user guide
########################################################################################################################
def openguide(CurrentDir):
    subprocess.Popen([CurrentDir + '\static\iCorrVision-2D.pdf'], shell=True)

########################################################################################################################
# Function to select the folder where the captured images will be saved
########################################################################################################################
def folder(console, folder_status, selectionFolder, k_snapshot):
    global filename_selection, preview_test

    filename_selection = filedialog.askdirectory()
    selectionFolder.set(filename_selection)

    if not filename_selection:
        folder_status.configure(bg='red')  # Red indicator
        console.insert(tk.END, 'The captured folder was not selected\n\n')
        console.see('insert')
        messagebox.showerror('Error', 'The captured folder was not selected!')
    else:
        if len(os.listdir(selectionFolder.get())) != 0:
            ans = messagebox.askquestion('Overwrite', 'Are you sure you want to overwrite the acquired images?',
                                         icon='question')
            if ans == 'yes':
                for f in os.listdir(selectionFolder.get()):
                    os.remove(selectionFolder.get() + '\\' + f)

                folder_status.configure(bg='#00cd00')  # Green indicator
                console.insert(tk.END, f'Image captured folder - {selectionFolder.get()}\n\n')
                console.see('insert')

                k_snapshot.set(1)

            else:
                folder_status.configure(bg='red')  # Red indicator
                console.insert(tk.END, 'The captured folder was not selected\n\n')
                console.see('insert')
                messagebox.showerror('Error', 'The captured folder was not selected!')

        else:
            for f in os.listdir(selectionFolder.get()):
                os.remove(selectionFolder.get() + '\\' + f)

            folder_status.configure(bg='#00cd00')  # Green indicator
            console.insert(tk.END, f'Image captured folder - {selectionFolder.get()}\n\n')
            console.see('insert')

            k_snapshot.set(1)

########################################################################################################################
# Preview images starter
########################################################################################################################
def preview(menu, preview_test, visualize_status, console, Width, Height, Exposure, Fps, amplifyImg,
            canvas_img, gridvar1, gridvar2):
    global cancel_id

    if preview_test.get():
        menu.after_cancel(cancel_id)
        preview_test.set(False)

    preview_test.set(True)
    visualize_status.configure(bg='#00cd00')  # Green indicator
    video_start(console, visualize_status, Width, Height, Exposure, Fps)
    t2 = Thread(target=preview_func, args=(menu, amplifyImg, canvas_img, gridvar1, gridvar2))
    t2.setDaemon(True)
    t2.start()

########################################################################################################################
# Function to start the camera
########################################################################################################################
def video_start(console, visualize_status, Width, Height, Exposure, Fps):
    global cameras, converter, cam, devices, tlFactory, cam_on

    tlFactory = pylon.TlFactory.GetInstance()

    devices = tlFactory.EnumerateDevices()

    if len(devices) == 0:
        console.insert(tk.END,
                       'No cameras were found. Make sure the cameras are connected to the computer\'s USB 3.0 ports!\n\n')
        console.see('insert')
        messagebox.showerror('Error',
                             'No cameras were found. Make sure the cameras are connected to the computer\'s USB 3.0 ports!')
        visualize_status.configure(bg='#ffa500')

    cameras = pylon.InstantCameraArray(min(len(devices), 1))

    l = cameras.GetSize()

    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(devices[i]))

        # Camera parameters:
        cam.Open()
        cam.Width.SetValue(Width.get())
        cam.Height.SetValue(Height.get())
        cam.ExposureTime.SetValue(Exposure.get())
        cam.AcquisitionFrameRate.SetValue(Fps.get())

        console.insert(tk.END,
                       f'Camera {cam.GetDeviceInfo().GetModelName()} - Serial number {cam.GetDeviceInfo().GetSerialNumber()} \n Temperature: {cam.DeviceTemperature.GetValue()} \n\n')
        console.see('insert')

        cam.Close()

    cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()

    # converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    # converter.OutputPixelFormat = pylon.PixelType_Mono10packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    startvideo = True

########################################################################################################################
# Preview function
########################################################################################################################
def preview_func(menu, amplifyImg, canvas_img, gridvar1, gridvar2):
    global cameras, converter, cancel_id, image0, image1

    grabResult0 = cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if cameras.IsGrabbing():

        image0 = converter.Convert(grabResult0)
        frame1 = cv.resize(image0.GetArray(), (640, 480))

        if amplifyImg.get():
            dxfigure = int(image0.GetArray().shape[1])
            dyfigure = int(image0.GetArray().shape[0])

            screen_height = menu.winfo_screenheight() - 100

            amplify_img = cv.resize(image0.GetArray(), (int(screen_height * dxfigure / dyfigure), screen_height))
            cv.imshow('Amplify image - Press q to close window', amplify_img)

        if cv.waitKey(1) == ord('q'):
            amplifyImg.set(False)
            cv.destroyAllWindows()

        if gridvar2.get() == 1:
            cv.line(frame1, (int(0.1 * 640), 0), (int(0.1 * 640), 480), (100, 255, 0), 1)
            cv.line(frame1, (int(0.2 * 640), 0), (int(0.2 * 640), 480), (100, 255, 0), 1)
            cv.line(frame1, (int(0.3 * 640), 0), (int(0.3 * 640), 480), (100, 255, 0), 1)
            cv.line(frame1, (int(0.4 * 640), 0), (int(0.4 * 640), 480), (100, 255, 0), 1)
            cv.line(frame1, (int(0.6 * 640), 0), (int(0.6 * 640), 480), (100, 255, 0), 1)
            cv.line(frame1, (int(0.7 * 640), 0), (int(0.7 * 640), 480), (100, 255, 0), 1)
            cv.line(frame1, (int(0.8 * 640), 0), (int(0.8 * 640), 480), (100, 255, 0), 1)
            cv.line(frame1, (int(0.9 * 640), 0), (int(0.9 * 640), 480), (100, 255, 0), 1)
            cv.line(frame1, (0, int(0.1 * 480)), (640, int(0.1 * 480)), (100, 255, 0), 1)
            cv.line(frame1, (0, int(0.2 * 480)), (640, int(0.2 * 480)), (100, 255, 0), 1)
            cv.line(frame1, (0, int(0.3 * 480)), (640, int(0.3 * 480)), (100, 255, 0), 1)
            cv.line(frame1, (0, int(0.4 * 480)), (640, int(0.4 * 480)), (100, 255, 0), 1)
            cv.line(frame1, (0, int(0.6 * 480)), (640, int(0.6 * 480)), (100, 255, 0), 1)
            cv.line(frame1, (0, int(0.7 * 480)), (640, int(0.7 * 480)), (100, 255, 0), 1)
            cv.line(frame1, (0, int(0.8 * 480)), (640, int(0.8 * 480)), (100, 255, 0), 1)
            cv.line(frame1, (0, int(0.9 * 480)), (640, int(0.9 * 480)), (100, 255, 0), 1)
            cv.line(frame1, (int(640 / 2), 0), (int(640 / 2), 480), (255, 0, 0), 1)
            cv.line(frame1, (0, int(480 / 2)), (640, int(480 / 2)), (255, 0, 0), 1)

        if gridvar1.get() == 1:
            cv.line(frame1, (int(640 / 2), 0), (int(640 / 2), 480), (255, 0, 0), 1)
            cv.line(frame1, (0, int(480 / 2)), (640, int(480 / 2)), (255, 0, 0), 1)

        img11 = ImageTk.PhotoImage(Image.fromarray(frame1))

        canvas_img.img11 = img11
        canvas_img.configure(image=img11)

    cancel_id = menu.after(2, preview_func, menu, amplifyImg, canvas_img, gridvar1, gridvar2)

########################################################################################################################
# Function to start the image acquisition
########################################################################################################################
def start_capture(menu, console, visualize_status, preview_test, Width, Height, Exposure, Fps, selectionFolder, TypeImg,
                  canvas_img, nImagesInit, TimeInit, TimeAcquisition):
    global running, k, start, cancel_id, capture_test

    console.delete('1.0', END)

    console.insert(tk.END, 'Image acquisition started\n\n')
    console.see('insert')
    visualize_status.configure(bg='red')  # Red indicator
    if preview_test.get():
        menu.after_cancel(cancel_id)
    k = 1
    running = True
    video_start(console, visualize_status, Width, Height, Exposure, Fps)

    start = time.time()
    Thread(target=Image_Capture, args=(menu, console, selectionFolder, TypeImg, canvas_img, nImagesInit, TimeInit, TimeAcquisition)).start()

########################################################################################################################
# Image acquisition function
########################################################################################################################
def Image_Capture(menu, console, selectionFolder, TypeImg, canvas_img, nImagesInit, TimeInit, TimeAcquisition):
    global running, k, start, cancel_capture, converter, cameras

    if running:

        img_name0 = selectionFolder.get() + f'\Image{k}{TypeImg.get()}'

        grabResult0 = cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        cv.imwrite(img_name0, converter.Convert(grabResult0).GetArray())

        img1 = ImageTk.PhotoImage(Image.fromarray(cv.resize(converter.Convert(grabResult0).GetArray(), (640, 480))))

        canvas_img.img1 = img1

        canvas_img.configure(image=img1)

        if k < nImagesInit.get():
            time.sleep(TimeInit.get() - (time.time() - start) % TimeInit.get())
        else:
            time.sleep(TimeAcquisition.get() - (time.time() - start) % TimeAcquisition.get())

        frame = time.time()

        console.insert(tk.END, f'Image: {k} Capture time: {frame - start}\n')
        console.see('insert')

        k = k + 1

    cancel_capture = menu.after(5, Image_Capture, menu, console, selectionFolder, TypeImg, canvas_img, nImagesInit, TimeInit, TimeAcquisition)

########################################################################################################################
# Function to stop the image capture
########################################################################################################################
def stop_capture(menu, console, selectionFolder, folder_status):
    global running
    global k, cancel_capture
    console.insert(tk.END, 'Image acquisition interrupted \n\n')
    console.see('insert')
    console.insert(tk.END, f'{k - 1} images were saved in {selectionFolder.get()} \n\n')
    console.see('insert')
    running = False
    menu.after_cancel(cancel_capture)
    folder_status.configure(bg='#ffa500')  # Orange indicator
    messagebox.showwarning('Attention',
                           'Please do not forget to modify the folder where the images of the next test will be saved!')

########################################################################################################################
# Function to close the program
########################################################################################################################
def close(menu):
    global cameras
    ans = messagebox.askquestion('Close', 'Are you sure you want to exit iCorrVision-2D Grabber module?',
                                 icon='question')
    if ans == 'yes':
        menu.destroy()
        menu.quit()

########################################################################################################################
# Function to capture snapshots
########################################################################################################################
def snapshot(console, selectionFolder, preview_test, k_snapshot, TypeImg):
    global converter, cameras, grabResult0

    # Current directory:
    CurrentDir = os.path.dirname(os.path.realpath(__file__))

    if preview_test.get():

        img_name0 = selectionFolder.get() + f'\Image{k_snapshot.get()}{TypeImg.get()}'

        grabResult0 = cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if cameras.IsGrabbing():
            image0 = converter.Convert(grabResult0)

            cv.imwrite(img_name0, image0.GetArray())

        console.insert(tk.END, f'Snapshot {k_snapshot.get()} . . .\n')
        console.see('insert')

        k_snapshot.set(k_snapshot.get() + 1)

    else:
        messagebox.showerror('Error', 'Before taking the snapshot, initialize the cameras!')

########################################################################################################################
# Function to designate image amplification
########################################################################################################################
def amplifyI(preview_test, amplifyImg):

    if preview_test.get():

        amplifyImg.set(True)

    else:
        messagebox.showerror('Error', 'First click on the preview button!')

