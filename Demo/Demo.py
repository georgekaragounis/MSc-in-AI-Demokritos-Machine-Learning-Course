from tkinter import  *
import joblib
from tkinter import messagebox
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import pickle
import cv2

from sklearn.preprocessing import StandardScaler

svm_rbf = pickle.load(open('svm.pkl', 'rb'))
scaler = StandardScaler()

root = Tk()

root.title("Pneumonia Predictor")

root.geometry("600x400")


root.config(bg="lightblue")





def classify_button_click():
    pX_ui = []
    file = filedialog.askopenfilename()
    if  file:
            image = cv2.imread(file)
            feature_matrix = np.zeros((64,64)) 
            #resize images to 64 x 64 pixels
            image_array = cv2.resize(image , (64,64))
            for i in range(0,image_array.shape[0]):
                for j in range(0,image_array.shape[1]):
                    feature_matrix[i][j] = ((int(image_array[i,j,0]) + int(image_array[i,j,1]) + int(image_array[i,j,2]))/3)

            image_ar = np.reshape(feature_matrix, (64*64)) 
            pX_ui.append(list(image_ar))
            pX_scaledui = scaler.fit_transform(pX_ui)
            pred = svm_rbf.predict(pX_scaledui)
            if pred == 0:
                vp="NORMAL"
            else:
                vp="PNEUMONIA"
    result_text.delete('1.0', tk.END)
    result_text.insert(tk.END, f"{vp}\n") 
    img = ImageTk.PhotoImage(Image.open(file).resize((100, 100)))
    image_label.config(image=img)
    image_label.image = img 
      
header = Label(root,text="Pneumonia Predictor",bg="lightblue",
     foreground="black",font=("Arial",15,"bold"))
header.pack()

frame1= Frame(root,bg="lightblue")
frame1.pack()

label1 = Label(frame1,text=" Chest X-Ray Image",bg="lightblue",foreground="black",
     font=("Arial",15,"bold"))
label1.grid(row=0,column=0,pady=10)

image_label = tk.Label(frame1,bg="lightblue")

image_label.grid(row=1,column=0,pady=2)

result_text = tk.Text(root,width=10, height=2)

browse_button = tk.Button(root, text="Browse", command=classify_button_click)



result_text.pack(pady=2)
browse_button.pack(pady=10)










root.mainloop()