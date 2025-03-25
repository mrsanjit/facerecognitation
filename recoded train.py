import tkinter as tk
from tkinter import Message, Text
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

# Create the main window
window = tk.Tk()
window.title("Face_Recogniser Attendance System")
window.geometry('1366x768')
window.configure(background='grey')

# Window configuration
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

# Display images
path = "im0.jpg"
path2 = "im1.jpg"
img = ImageTk.PhotoImage(Image.open(path))
img2 = ImageTk.PhotoImage(Image.open(path2))
panel = tk.Label(window, image=img)
panel2 = tk.Label(window, image=img2)
panel.pack(side="left", fill="x", expand="no")
panel2.pack(side="left", fill="x", expand="no")

message = tk.Label(window, text="Face-Recognition-Attendance-System", bg="grey",
                   fg="black", width=50, height=3, font=('arial', 30, 'italic bold underline'))
message.place(x=80, y=20)

# Labels and Entry Widgets
lbl = tk.Label(window, text="EMP ID", width=20, height=2,
               fg="white", bg="green", font=('times', 15, ' bold '))

lbl.place(x=400, y=200)

txt = tk.Entry(window, width=20, bg="green",
               fg="white", font=('times', 15, ' bold '))

txt.place(x=700, y=215)

lbl2 = tk.Label(window, text="Employee Name", width=20, fg="white",
                bg="green", height=2, font=('times', 15, ' bold '))

lbl2.place(x=400, y=300)

txt2 = tk.Entry(window, width=20, bg="green",
                fg="white", font=('times', 15, ' bold '))

txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Notification : ", width=20, fg="white",
                bg="green", height=2, font=('times', 15, ' bold underline '))

lbl3.place(x=400, y=400)

message = tk.Label(window, text="", bg="green", fg="white", width=30,
                   height=2, activebackground="yellow", font=('times', 15, ' bold '))

message.place(x=700, y=400)

lbl3 = tk.Label(window, text="Attendance : ", width=20, fg="white",
                bg="green", height=2, font=('times', 15, ' bold  underline'))

lbl3.place(x=400, y=600)

message2 = tk.Label(window, text="", fg="white", bg="green",
                    activeforeground="green", width=30, height=2, font=('times', 15, ' bold '))

message2.place(x=700, y=600)

# Helper Functions
def clear():
    txt.delete(0, 'end')
    message.configure(text="")

def clear2():
    txt2.delete(0, 'end')
    message.configure(text="")

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def TakeImages():
    Id = (txt.get())
    name = (txt2.get())
    
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            message.configure(text="Error: Could not access the camera.")
            return
        
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        
        if not detector.empty():
            sampleNum = 0
            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sampleNum += 1
                    cv2.imwrite(f"TrainingImage/{name}.{Id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
                    cv2.imshow('frame', img)

                if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 100:
                    break

            cam.release()
            cv2.destroyAllWindows()
            res = f"Images Saved for ID : {Id} Name : {name}"
            row = [Id, name]
            
            if not os.path.exists('EmployeeDetails'):
                os.makedirs('EmployeeDetails')
            
            with open('EmployeeDetails/EmployeeDetails.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            message.configure(text=res)
        else:
            message.configure(text="Error: Haar cascade not loaded.")
    else:
        if is_number(Id):
            res = "Enter Alphabetical Name"
        else:
            res = "Enter Numeric Id"
        message.configure(text=res)

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    res = "Image Trained"
    message.configure(text=res)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("EmployeeDetails/EmployeeDetails.csv")
    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    while True:
        ret, im = cam.read()
        if not ret:
            break  # If no frame is captured, break the loop

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                if len(aa) > 0:
                    name = aa[0]  # Extract the name (first element of the list)
                    tt = str(Id) + "-" + name
                    attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
                else:
                    name = "Unknown"
                    tt = str(Id)

            else:
                Id = 'Unknown'
                tt = str(Id)

            if conf > 75:
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite(f"ImagesUnknown/Image{noOfFile}.jpg", im[y:y + h, x + w])

            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)

        # Drop duplicates to ensure only one entry per person per session
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        
        # Display live frame with attendance information
        cv2.imshow('Face Recognition Attendance', im)

        # Stop the process when 'q' is pressed or if attendance is recorded
        if cv2.waitKey(1) & 0xFF == ord('q') or not attendance.empty:
            break

    # Save the attendance to a CSV file
    if not attendance.empty:
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour, Minute, Second = timeStamp.split(":")
        fileName = f"Attendance/Attendance_{date}_{Hour}-{Minute}-{Second}.csv"
        attendance.to_csv(fileName, index=False)
    
    # Release the camera and close windows
    cam.release()
    cv2.destroyAllWindows()

    # Update UI message
    message2.configure(text="Attendance has been marked!")
    # message2.configure(text=str(attendance.tail(5)))

    # Ready for quit action
    quitWindow.config(state=tk.NORMAL)  # Enable the quit button

# Create necessary directories if not exist
if not os.path.exists('TrainingImage'):
    os.makedirs('TrainingImage')

if not os.path.exists('TrainingImageLabel'):
    os.makedirs('TrainingImageLabel')

if not os.path.exists('ImagesUnknown'):
    os.makedirs('ImagesUnknown')

if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

if not os.path.exists('EmployeeDetails'):
    os.makedirs('EmployeeDetails')

# Buttons for the functions
clearButton = tk.Button(window, text="Clear", command=clear, fg="red", bg="yellow",
                        width=20, height=2, activebackground="Red", font=('times', 15, ' bold '))

clearButton.place(x=950, y=200)

clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="red", bg="yellow",
                         width=20, height=2, activebackground="Red", font=('times', 15, ' bold '))

clearButton2.place(x=950, y=300)

takeImg = tk.Button(window, text="Take Images", command=TakeImages, fg="red", bg="yellow",
                    width=20, height=3, activebackground="Red", font=('times', 15, ' bold '))

takeImg.place(x=200, y=500)

trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="red",
                     bg="yellow", width=20, height=3, activebackground="Red", font=('times', 15, ' bold '))

trainImg.place(x=500, y=500)

trackImg = tk.Button(window, text="Track Images", command=TrackImages, fg="red",
                     bg="yellow", width=20, height=3, activebackground="Red", font=('times', 15, ' bold '))

trackImg.place(x=800, y=500)

quitWindow = tk.Button(window, text="Quit", command=window.quit, fg="red", bg="yellow",
                       width=20, height=3, activebackground="Red", font=('times', 15, ' bold '))

quitWindow.place(x=1100, y=500)

# Initially disable the quit button
quitWindow.config(state=tk.DISABLED)

# Run the main window loop
window.mainloop()
