import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer as lbc
from tkinter import *
from tkinter import messagebox

data = lbc()
clm = np.array(data['feature_names'])
df_x = pd.DataFrame(data['data'])
df_y = pd.DataFrame(data['target'])
X = df_x.iloc[:, :].values
y = df_y.iloc[:, 0].values

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(X_train, y_train)

def about():
    rt_2 = Tk()
    rt_2.geometry('250x60')
    rt_2.title('About')
    dis = 'Author : Javidh\n Last edited : sep 5 2021'
    Label(rt_2, text=dis).pack()
    rt_2.mainloop()
def acc_pred():
    a = accuracy_score(y_test, reg.predict(X_test))*100
    dis = ('%.3f' % (float(a)))+'%'
    rt_1 = Tk()
    rt_1.geometry('250x60')
    rt_1.title('Accuracy')
    Label(rt_1, text = dis, font = ("Californian FB", 20)).pack()
    rt_1.mainloop()

def pred_1(y_pred):
    if reg.predict(y_pred) == [0]:
        dig = 'Malignant'
    else:
        dig = 'Benign'    
    rtd = Tk()
    rtd.geometry('400x60')
    rtd.title('Diognisis')
    Label(rtd, text = dig, font = ("Californian FB", 25)).pack()
    rtd.mainloop()
def pred(rt, btn, vls):
    vls.clear()
    for ind,x in enumerate(btn):
        try:
            val = float(x.get())
            vls.append(val)
        except:
            if clm[ind] in clm[10:20]:
                vls.append(0.0)
            else:
                messagebox.showerror('Invalid Formate', 'Not expected formate in '+str(clm[ind]))
                return
    y_pred = np.array(vls).reshape(1, len(vls))
    pred_1(y_pred)
def gui():
    rt  = Tk()
    rt.geometry('400x700')
    rt.title('cancer')
    btn = []
    vls = []
    mb = Menu(rt)
    file = Menu(mb, tearoff = 0)
    mb.add_cascade(label ='File', menu = file)
    file.add_command(label ='Accuracy', command = acc_pred)
    file.add_command(label ='About', command = about)
    rt.config(menu = mb)
    for x,y in enumerate(clm):
        Label(rt, text = y).grid(row = x, column = 0)
        a = Entry(rt, width = 10, )
        a.grid(row = x, column = 1)
        btn.append(a)
        if x in range(10,20):
            Label(rt, text = '#Default value: 0').grid(row = x, column = 2)
    Button(rt, text = 'Predict', command = lambda:pred(rt, btn, vls), bg = '#6C5959',activebackground = 'green').grid(row = x+1, column = 1)
    rt.mainloop()
gui()
