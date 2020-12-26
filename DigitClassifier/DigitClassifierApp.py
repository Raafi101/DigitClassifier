#GUI for Digit Classification

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn import *
from tkinter import *
import tkinter.font as tkfont
from cv2 import cv2
from PIL import Image, ImageDraw, ImageTk, ImageGrab, EpsImagePlugin
EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs9.53.3\bin\gswin64c'

#Load Neural Network Model
model = keras.models.load_model('DigitNNModel')

#Data =========================================================================

currentX, currentY = 0, 0

def locateXY(event):
    global currentX, currentY
    currentX, currentY = event.x, event.y
    print(currentX, currentY)

def paint(event):
    global currentX, currentY
    canvas.create_line((currentX, currentY, event.x, event.y), fill= 'black', width = 30, capstyle = ROUND, smooth = TRUE, stipple = '')
    currentX, currentY = event.x, event.y

counter = 0

def guessDigit():
    global counter
    canvas.postscript(file = 'userDigitEPS.eps', colormode='color')
    image = Image.open('userDigitEPS.eps')
    image.save('userDigitPNG.png', 'png')
    image = cv2.imread('userDigitPNG.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.bitwise_not(image)
    image = cv2.resize(image, (28,28))
    imageFlat = image.reshape(-1, 784) / 255
    pred = np.argmax(model.predict(imageFlat))
    yourNum['text'] = 'You drew a {0}'.format(pred)
    if counter == 0:
        yourNum.pack(pady = 50)
    counter += 1

def clearCanvas():
    canvas.delete('all')

#GUI Stuffs ===================================================================

root = Tk()
root['bg'] = 'light gray'
root.title('Digit Classifier')
root.geometry("1280x720")

fontStyle = tkfont.Font(family="Trebuchet", 
                        size=20
)

fontStyle2 = tkfont.Font(family="Trebuchet", 
                        size=14
)

fontStyle3 = tkfont.Font(family="Trebuchet", 
                        size=24
)

logoFrame = Frame(root)
logoFrame.config(background='light gray')
logoFrame.pack(side = TOP)
title = Label(logoFrame, 
              text = 'Digit Classification using AI', 
              bg = 'light gray', 
              fg = '#395099', 
              font = fontStyle, 
              pady = 20
)

titlePt2 = Label(logoFrame, 
              text = 'By Raafi Rahman', 
              bg = 'light gray', 
              fg = '#395099', 
              font = fontStyle2, 
              pady = 0
)

title.pack()
titlePt2.pack()

canvasFrame = Frame(root,
                    pady = 15,
                    bg = 'light gray'
)

canvasFrame.config(background='light gray')

descriptionFrame = Frame(canvasFrame)

descriptionFrame.config(background='light gray')

description = Message(descriptionFrame,
                    text = 'Draw a digit on the canvas and press "Guess" to run the AI. Make sure you draw your digit large and clearly',
                    font = fontStyle2,
                    anchor = CENTER,
                    width = 500,
                    bg = 'light gray',
                    fg = '#395099'
)

descriptionFrame.pack(side = LEFT,
                      fill = X
)

description.pack(side = LEFT)

canvas = Canvas(canvasFrame,
                width = 300,
                height = 300,
                bg='white'
)

canvasFrame.pack(fill = X,
                 padx = 100
)

canvas.pack(side = RIGHT)

buttonFrame = Frame(descriptionFrame)
buttonFrame.config(background='light gray')
buttonFrame.pack(side = RIGHT)

numFrame = Frame(root)
numFrame.config(background='light gray')
numFrame.pack(side = BOTTOM)
yourNum = Label(numFrame,
                font = fontStyle3,
                bg = 'light gray',
                fg = '#395099')


#On click
canvas.bind('<Button-1>', locateXY)

#On drag
canvas.bind('<B1-Motion>', paint)

guess = Button(buttonFrame)
guess['text'] = 'Guess'
guess['command'] = guessDigit
guess.pack()

clear = Button(buttonFrame)
clear['text'] = 'Clear Canvas'
clear['command'] = clearCanvas
clear.pack()

mainloop()