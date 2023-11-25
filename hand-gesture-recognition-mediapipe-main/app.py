
# -*- coding: utf-8 -*-
# second.deiconify()
import argparse
import copy
import csv
import itertools
from collections import Counter
from collections import deque
import pyttsx3
#import PyAudio
import cv2 as cv
import mediapipe as mediapipe
import numpy as numpy
from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier

import pyodbc
from tkinter import *
from tkinter import messagebox
from tkinter import *
import customtkinter
customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"
import tkinter as tk
from PIL import Image, ImageTk
from itertools import count, cycle

import speech_recognition as sr

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()
        root.quit()

r = sr.Recognizer()

class ImageLabel(tk.Label):
    def load(self, image):
        if isinstance(image, str):
            im = Image.open(image)
        frames = []

        try:
            for i in count(1):
                frames.append(ImageTk.PhotoImage(image.copy()))
                image.seek(i)
        except EOFError:
            pass
        self.frames = cycle(frames)

        try:
            self.delay = image.info['duration']
        except:
            self.delay = 100

        if len(frames) == 1:
            self.config(image=next(self.frames))
        else:
            self.next_frame()

    def unload(self):
        self.config(image=None)
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.config(image=next(self.frames))
            self.after(self.delay, self.next_frame)


def on_closingtexttosignw():
    texttosignwinn.withdraw()
    top.deiconify()
def speechtoSignwin():
    # speechtoSignWin = customtkinter.CTk()
    speechtoSignWin = Toplevel(root)
    speechtoSignWin.title("Speech to Sign")

    speechtoSignWin.geometry("500x500")
    speechtoSignWin.configure(bg="grey")

    def on_closingspeechtoSignw():
        speechtoSignWin.withdraw()
        top.deiconify()




    speechtoSignWin.protocol("WM_DELETE_WINDOW", on_closingspeechtoSignw)
    # textToConvertLabel = Label(speechtoSignwin, text='Hello There!', bg='grey', fg='white', font=('Arial', 20))
    # textToConvertLabel.place(x=175, y=20)
    # textToConvertEntry = Entry(speechtoSignwin, font=('Arial', 20))
    # textToConvertEntry.place(x=120, y=400)

    def converttosign():
        with sr.Microphone() as source:
            print("say something...")
            # read the audio data from the default microphone
            audio_data = r.record(source, duration=5)
            print("Recognizing...")
            # convert speech to text
            text = r.recognize_google(audio_data)
            # print(text)

            lbl2.unload()
            entry = text
            str1 = ".gif"
            newEntry = entry + str1

            lbl2.pack()
            lbl2.load(newEntry)
    # btn_sign_spch = Button(speechtoSignWin, text='Record', bg='grey', fg='white', font=('Arial', 20), bd=5,
    #                        command=converttosign)
    # btn_sign_spch.place(x=170, y=450)
    btn_sign_spch = customtkinter.CTkButton(master=speechtoSignWin, text="Record", command=converttosign)
    btn_sign_spch.place(relx=0.5, rely=0.9, anchor=CENTER)
    lbl2 = ImageLabel(speechtoSignWin)
def texttosignwin():


    texttosignwinn.deiconify()
    texttosignwinn.protocol("WM_DELETE_WINDOW", on_closingtexttosignw)
    # textToConvertEntry = Entry(texttosignwin, font=('Arial', 20))
    # textToConvertEntry.place(x=120, y=400)

    textToConvertEntry = customtkinter.CTkEntry(master=texttosignwinn, placeholder_text="Type Text here", width=250)
    textToConvertEntry.place(relx=0.5, rely=0.8, anchor=CENTER)
    # entry2 = Entry(root, font=('Arial', 20), show='*')
    # entry2.place(x=170, y=180)

    # button = Button(root, text='Login', bg='grey', fg='white', font=('Arial', 20), bd=5, command=login)

    lbl = ImageLabel(texttosignwinn)
    def convert():
        lbl.unload()
        entry = textToConvertEntry.get()
        str1 = ".gif"
        newEntry = entry+str1

        lbl.pack()
        lbl.load(newEntry)
    # btn_sign_spch = Button(texttosignwin, text='Convert', bg='grey', fg='white', font=('Arial', 20), bd=5,command=convert)
    # btn_sign_spch.place(x=170, y=450)

    btn_sign_spch = customtkinter.CTkButton(master=texttosignwinn, text="Convert", command=convert)
    btn_sign_spch.place(relx=0.5, rely=0.9, anchor=CENTER)




def on_closing_pass():
    chgpaswrd.withdraw()
    top.deiconify()
def on_closing_newuser():
    adduser.withdraw()
    top.deiconify()

def adduder():
    top.withdraw()
    adduser.deiconify()
    label11 = customtkinter.CTkLabel(master=adduser, text="New User",
                                    font=customtkinter.CTkFont(size=20, weight="bold"))
    label11.place(relx=0.5, rely=0.1, anchor=CENTER)

    isactive = True
    entrynewUsername = customtkinter.CTkEntry(master=adduser, placeholder_text="UserName", width=250)
    entrynewUsername.place(relx=0.5, rely=0.3, anchor=CENTER)
    # entry1.place(x=170, y=100)

    entrynewsurname = customtkinter.CTkEntry(master=adduser, placeholder_text="Surname", width=250,)
    entrynewsurname.place(relx=0.5, rely=0.4, anchor=CENTER)

    entrynewNewPAass = customtkinter.CTkEntry(master=adduser, placeholder_text="Password", width=250, show='*')
    entrynewNewPAass.place(relx=0.5, rely=0.5, anchor=CENTER)

    entrynewNewPAassAgain = customtkinter.CTkEntry(master=adduser, placeholder_text="Retype Password", width=250,
                                                show='*')
    entrynewNewPAassAgain.place(relx=0.5, rely=0.6, anchor=CENTER)
    def Add_new_user():
        newusername = entrynewUsername.get()

        newsurname= entrynewsurname.get()
        newuserpassord = entrynewNewPAass.get()
        newuserpassordagain = entrynewNewPAassAgain.get()
        if newusername == '' or newsurname == '' or newuserpassord == '' or newuserpassordagain == '':
            message = "Please fill in all Field"
            messagebox.showerror('Error', message)
        elif newuserpassord != newuserpassordagain:
            message = "New passwords do not match"
            messagebox.showerror('Error', message)
        else:
            DRIVER = 'SQL SERVER'
            SERVER = 'DESKTOP-8KUKLPJ'
            DATABASE = 'SignLanguage'
            connectionString = f'DRIVER={{SQL Server}};SERVER={SERVER};DATABASE={DATABASE};'
            conn = pyodbc.connect(connectionString)
            cursor = conn.cursor()
            cursor.execute("""
                                        INSERT INTO [SignLanguage].[dbo].[Users] (UserName, Surname,Password,IsActive)
                                        VALUES (?, ?, ?, 1)                         
                                        """, [newusername,newsurname, newuserpassord])
            cursor.commit()
            message = "New User Added"
            adduser.withdraw()
            top.deiconify()
            messagebox.showinfo('Success', message)


    buttonnewvalidate = customtkinter.CTkButton(master=adduser, text="Add User", command=Add_new_user)
    buttonnewvalidate.place(relx=0.5, rely=0.9, anchor=CENTER)

    adduser.protocol("WM_DELETE_WINDOW", on_closing_newuser)

def changepassword():
    top.withdraw()
    chgpaswrd.deiconify()
    label1 = customtkinter.CTkLabel(master=chgpaswrd, text="Hello There",font=customtkinter.CTkFont(size=20, weight="bold"))
    label1.place(relx=0.5, rely=0.1, anchor=CENTER)


    entryUsername = customtkinter.CTkEntry(master=chgpaswrd, placeholder_text="UserName",width=250)
    entryUsername.place(relx=0.5, rely=0.3, anchor=CENTER)
    # entry1.place(x=170, y=100)

    entryOLdPAass = customtkinter.CTkEntry(master=chgpaswrd, placeholder_text="Old Password", width=250,show='*')
    entryOLdPAass.place(relx=0.5, rely=0.4, anchor=CENTER)

    entryNewPAass = customtkinter.CTkEntry(master=chgpaswrd, placeholder_text="New Password", width=250, show='*')
    entryNewPAass.place(relx=0.5, rely=0.5, anchor=CENTER)

    entryNewPAassAgain = customtkinter.CTkEntry(master=chgpaswrd, placeholder_text="Retype New Password", width=250, show='*')
    entryNewPAassAgain.place(relx=0.5, rely=0.6, anchor=CENTER)




    # entry2 = Entry(root, font=('Arial', 20), show='*')
    # entry2.place(x=170, y=180)

    # button = Button(root, text='Login', bg='grey', fg='white', font=('Arial', 20), bd=5, command=login)

    def Change_password():
        message = ""
        username = entryUsername.get()
        oldpassword = entryOLdPAass.get()
        newpassword = entryNewPAass.get()
        newpasswordAgain = entryNewPAassAgain.get()

        # ==================================
        DRIVER = 'SQL SERVER'
        SERVER = 'DESKTOP-8KUKLPJ'
        DATABASE = 'SignLanguage'
        connectionString = f'DRIVER={{SQL Server}};SERVER={SERVER};DATABASE={DATABASE};'
        conn = pyodbc.connect(connectionString)
        cursor = conn.cursor()
        cursor.execute("""
                    SELECT TOP (1000) [Id]
                      ,[UserName]
                      ,[Surname]
                      ,[Password]
                      ,[IsActive]
                    FROM [SignLanguage].[dbo].[Users]
                    WHERE Username = ? AND password = ?
                    """, [username, oldpassword])
        records = cursor.fetchall()

        # ===================================
        if username == '' or oldpassword == '' or oldpassword == '' or newpassword == '' or newpasswordAgain == '' :
            message = "Please fill in all Field"
            messagebox.showerror('Error', message)
        elif newpassword != newpasswordAgain:
            message = "New passwords do not match"
            messagebox.showerror('Error', message)
        elif (records):
            id = records[0].Id
            if f"{records[0].IsActive}" == 'True':
                #==============================================================
                DRIVER = 'SQL SERVER'
                SERVER = 'DESKTOP-8KUKLPJ'
                DATABASE = 'SignLanguage'
                connectionString = f'DRIVER={{SQL Server}};SERVER={SERVER};DATABASE={DATABASE};'
                conn = pyodbc.connect(connectionString)
                cursor = conn.cursor()
                cursor.execute("""
                            UPDATE [SignLanguage].[dbo].[Users]
                            SET Password = ?
                            WHERE Id = ?                         
                            """, [newpassword, id])
                cursor.commit()

                #=================================================================
                message = "Thank your Password has been changed, Please Login"
                chgpaswrd.withdraw()
                root.deiconify()
                messagebox.showinfo('Success', message)

        else:
            message = "Wrong Credential or Not Active User"
            messagebox.showerror('Error', message)



    buttonvalidate = customtkinter.CTkButton(master=chgpaswrd,text="Update",command=Change_password)
    buttonvalidate.place(relx=0.5, rely=0.9, anchor=CENTER)

    chgpaswrd.protocol("WM_DELETE_WINDOW", on_closing_pass)


def login():
    username = entry1.get()
    password = entry2.get()
    if username == '' and password == '':
        message = "both field should be filled"
        messagebox.showerror('Login Error',message)
    elif username == '' or password == '':
        if username == '':
            message = "Please provide the username"
            messagebox.showerror('Login Error',message)
        elif password == '':
            message = "Please provide the password"
            messagebox.showerror('Login Error',message)
    elif username != '' and password != '':
        DRIVER = 'SQL SERVER'
        SERVER = 'DESKTOP-8KUKLPJ'
        DATABASE = 'SignLanguage'
        connectionString = f'DRIVER={{SQL Server}};SERVER={SERVER};DATABASE={DATABASE};'
        conn = pyodbc.connect(connectionString)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT TOP (1000) [id]
              ,[UserName]
              ,[Surname]
              ,[Password]
              ,[IsActive]
            FROM [SignLanguage].[dbo].[Users]
            WHERE Username = ? AND password = ?
            """, [username, password])
        records = cursor.fetchall()
        if(records):
            print(records[0].IsActive)
            if f"{records[0].IsActive}" == 'True':

                message = "'Hello {} {} \n Press Okay Or close the Window to start".format(records[0].UserName,records[0].Surname)
                messagebox.showinfo('Login Successful', message)
                entry1.delete(0,END)
                entry2.delete(0, END)
                root.withdraw()
                top.deiconify()
                # top = Toplevel(root)
                #
                # top.title("Options")
                # top.geometry("500x400")
                def on_closing_top():
                    top.withdraw()
                    root.deiconify()

                top.protocol("WM_DELETE_WINDOW", on_closing_top)






                # top.eval('tk::PlaceWindow . center')

                # label4 = Label(top, text='Welcome', bg='grey', fg='white', font=('Arial', 20))
                # label4.place(x=50, y=20)
                def speechtoSign():
                    top.withdraw()
                    speechtoSignwin()
                # btn_spch_sign = Button(top, text='Speech to Sign', bg='grey', fg='white', font=('Arial', 20), bd=5,command= speechtoSign )
                # btn_spch_sign.place(x=140, y=50)
                btn_spch_sign = customtkinter.CTkButton(master=top, text="Speech to Sign", command=speechtoSign)
                btn_spch_sign.place(relx=0.5, rely=0.1, anchor=CENTER)

                # btn_sign_spch = Button(top, text='Sign to Speech ', bg='grey', fg='white', font=('Arial', 20), bd=5, )
                # btn_sign_spch.place(x=140, y=120)
                def signtotext():
                    messagebox.showinfo('Login Success', 'Hello Lisette\n Press okay to start')
                    top.withdraw()
                    main()
                btn_spch_sign = customtkinter.CTkButton(master=top, text="Sign to Speech",command=signtotext)
                btn_spch_sign.place(relx=0.5, rely=0.3, anchor=CENTER)



                # btn_sign_txt = Button(top, text='Sign to Text ', bg='grey', fg='white', font=('Arial', 20), bd=5,
                #                       command=signtotext)
                # btn_sign_txt.place(x=160, y=190)
                btn_spch_sign = customtkinter.CTkButton(master=top, text="Sign to Text", command=signtotext)
                btn_spch_sign.place(relx=0.5, rely=0.5, anchor=CENTER)

                def texttosign():
                    top.withdraw()
                    texttosignwin()

                btn_spch_sign = customtkinter.CTkButton(master=top, text="Text to Sign", command=texttosign)
                btn_spch_sign.place(relx=0.5, rely=0.7, anchor=CENTER)
                # btn_chng_password = Button(top, text='Change Password', bg='grey', fg='white', font=('Arial', 20), bd=5, )
                # btn_chng_password.place(x=130, y=330)
                btn_spch_sign = customtkinter.CTkButton(master=top, text="Change Password",command=changepassword)
                btn_spch_sign.place(relx=0.3, rely=0.9, anchor=CENTER)

                btn_spch_sign = customtkinter.CTkButton(master=top, text="Add User", command=adduder)
                btn_spch_sign.place(relx=0.7, rely=0.9, anchor=CENTER)


                btn_back = customtkinter.CTkButton(master=top,fg_color="red",width=80, text="back",command = on_closing_top)
                btn_back.place(relx=0.0, rely=0.0)

            else:
                message = "Inactive User"
                messagebox.showerror('Login Error', message)
        else:
            message = "Wrong password or Username"
            messagebox.showerror('Login Error', message)
def loadArguments():
    analyzer = argparse.ArgumentParser()

    analyzer.add_argument("--device", type=int, default=0)
    analyzer.add_argument("--width", help='cap width', type=int, default=960)
    analyzer.add_argument("--height", help='cap height', type=int, default=560)# cahnged values here

    analyzer.add_argument('--use_static_image_mode', action='store_true')
    analyzer.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    analyzer.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    arguments = analyzer.parse_args()

    return arguments


def main():

    # Argument parsing #################################################################
    arguments = loadArguments()

    screenName = arguments.device
    screenWidth = arguments.width
    screenHeight = arguments.height

    use_static_image_mode = arguments.use_static_image_mode
    min_detection_confidence = arguments.min_detection_confidence
    min_tracking_confidence = arguments.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(screenName)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, screenWidth)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, screenHeight)

    # Model load #############################################################
    handsMediapipe = mediapipe.solutions.hands
    hands = handsMediapipe.Hands(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        static_image_mode=use_static_image_mode,
        max_num_hands=1,# change number of hands
        
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # get all the Labels
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # Frame fer seconds
    FPS_Calculation = CvFpsCalc(buffer_len=10)

    # CLast coordinate
    howFarHistory = 16
    point_history = deque(maxlen=howFarHistory)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=howFarHistory)

    #  ########################################################################
    modeSetup = 0
# as long as the screen is open
    while True:
        fps = FPS_Calculation.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC to close the window
            top.deiconify()
            break
        number, modeSetup = select_mode(key, modeSetup)
        # say the words 'J' if you want to say the words
        if key == 122:
            saythis= (keypoint_classifier_labels[keypoint_classifier(pre_processed_landmark_list)])
            say_it_in_words(saythis)
        # all about the camera
        ret, image = cap.read()# reads the frames
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)# convert

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                print(hand_landmarks)#a dded by emmanuel
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                #print(landmark_list)#added as well
                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write results to the dataset files
                logging_csv(number, modeSetup, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])# chenge point history here
                else:
                    point_history.append([0, 0])

                # fingers
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (howFarHistory * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing section
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        # debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, modeSetup, number)

        # Screen detection
        cv.imshow('Two-Way Sign Recognition System', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, modeSetup):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        modeSetup = 0
    if key == 107:  # k
        modeSetup = 1
    if key == 104:  # h
        modeSetup = 2
    return number, modeSetup


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = numpy.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [numpy.array((landmark_x, landmark_y))]

        landmark_array = numpy.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def say_it_in_words(wordsToSay):
    text_speech = pyttsx3.init()
    text_speech.say(wordsToSay)
    text_speech.runAndWait()


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, modeSetup, landmark_list, point_history_list):
    if modeSetup == 0:
        pass
    if modeSetup == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if modeSetup == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (100, 100, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (100, 100, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 100, 100), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 100, 100), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 100, 100), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (100, 255, 100), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (100, 255, 100), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (100, 255, 100), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 100, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 100, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 100, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (50, 50, 100), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (50, 50, 100), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (50, 50, 100), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  #
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    #uncomment to see text on screen
    if finger_gesture_text != "":
        cv.putText(image, "Hand Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.FONT_ITALIC)
        cv.putText(image, "Hand Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.FONT_ITALIC)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, modeSetup, number):
    cv.putText(image, "Signs -> Text ", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "Signs -> Text ", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 100, 255), 2, cv.LINE_AA)

    cv.putText(image, "Frame Rate:" + str(fps), (150, 460), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "Frame Rate:" + str(fps), (150, 460), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= modeSetup <= 2:
        cv.putText(image, "MODE:" + mode_string[modeSetup - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    root = customtkinter.CTk()
    root.title("SignItUp-Login")
    root.configure(bg='grey')

    top = Toplevel(root)
    top.title("Options")
    top.geometry("500x400")
    top.withdraw()


    chgpaswrd = Toplevel(root)
    chgpaswrd.title("change password")
    chgpaswrd.geometry("500x400")
    chgpaswrd.withdraw()

    adduser = Toplevel(root)
    adduser.title("Add User")
    adduser.geometry("500x400")
    adduser.withdraw()


    texttosignwinn = Toplevel(root)

    texttosignwinn.title("Text to Sign")
    # speechtoSignWin.configure(bg='grey')
    texttosignwinn.geometry("500x500")
    texttosignwinn.withdraw()
    global entry1
    global entry2

    root.geometry("500x400")
    # frame = customtkinter.CTkFrame(master=root, width=250, height=50)
    # frame.place(relx=0.5, rely=0.1)

    # label1 = Label(root, text='Hello There!', bg='grey', fg='white', font=('Arial', 20))
    # label1.place(x=175, y=20)

    label1 = customtkinter.CTkLabel(master=root, text="Hello There",font=customtkinter.CTkFont(size=20, weight="bold"))
    label1.place(relx=0.5, rely=0.1, anchor=CENTER)


    entry1 = customtkinter.CTkEntry(master=root, placeholder_text="UserName",width=250)
    entry1.place(relx=0.5, rely=0.3, anchor=CENTER)
    # entry1.place(x=170, y=100)

    entry2 = customtkinter.CTkEntry(master=root, placeholder_text="Password", width=250,show='*')
    entry2.place(relx=0.5, rely=0.4, anchor=CENTER)
    # entry2 = Entry(root, font=('Arial', 20), show='*')
    # entry2.place(x=170, y=180)

    # button = Button(root, text='Login', bg='grey', fg='white', font=('Arial', 20), bd=5, command=login)
    button = customtkinter.CTkButton(master=root,text="Login",command=login)
    button.place(relx=0.5, rely=0.6, anchor=CENTER)

    root.eval('tk::PlaceWindow . center')
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
#main()

