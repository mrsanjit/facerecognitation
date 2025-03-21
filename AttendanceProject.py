import dlib
import cv2
import numpy as np
import face_recognition
import os
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import csv

path = 'Images_Attendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Function to find face encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Function to mark attendance and track login/logout
def markLoginLogout(name, action):
    with open('login_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        time_now = datetime.now()
        time_string = time_now.strftime('%H:%M:%S')
        date_string = time_now.strftime('%d/%m/%Y')
        writer.writerow([name, action, time_string, date_string])

# Function for the login window
def showLoginWindow(name):
    window = tk.Tk()
    window.title("User Login")
    window.geometry("300x200")

    def on_logout():
        markLoginLogout(name, "Logout")
        messagebox.showinfo("Logged Out", f"{name} has logged out.")
        window.destroy()

    # Show the user's name and provide a logout button
    label = tk.Label(window, text=f"Welcome {name}!", font=("Arial", 14))
    label.pack(pady=20)

    logout_button = tk.Button(window, text="Logout", command=on_logout, font=("Arial", 12), bg="red", fg="white")
    logout_button.pack(pady=20)

    window.mainloop()

# Function to display the camera feed and handle login
def loginSystem():
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                markLoginLogout(name, "Login")
                showLoginWindow(name)  # Open the login window
                break  # Exit the loop after logging in

        cv2.imshow('webcam', img)

        if cv2.waitKey(10) == 13:  # Press Enter to exit the loop
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    loginSystem()
