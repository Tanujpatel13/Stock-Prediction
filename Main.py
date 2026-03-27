from tkinter import messagebox
from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tkinter import ttk

# UPDATED KERAS IMPORTS
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, LSTM, Dropout

import pickle

main = Tk()
main.title("Stock Price Prediction using LSTM & ANN")
main.geometry("1300x1200")

global filename
global dataset
global X, Y, mse, X_train, X_test, y_train, y_test
sc = MinMaxScaler(feature_range=(0,1))
global stock_name, stock_list


def uploadDataset():
    global filename, stock_name, dataset, stock_list
    text.delete('1.0', END)
    filename = askopenfilename(initialdir="Dataset")
    fname = os.path.basename(filename)
    stock_name = stock_list.get()

    if fname == 'NSE-Tata-Global-Beverages-Limited.csv':
        dataset = pd.read_csv(filename,usecols=['Date','Open','High','Low','Close'])
        dataset["Date"] = pd.to_datetime(dataset.Date,format="%Y-%m-%d")
        dataset.index = dataset['Date']
        dataset = dataset.sort_index(ascending=True, axis=0)
        dataset.fillna(0, inplace=True)
        stock_name = 'NSE-Tata-Global-Beverages-Limited'
    else:
        dataset = pd.read_csv(filename,usecols=['Date','Open','High','Low','Close','Stock'])
        dataset["Date"] = pd.to_datetime(dataset.Date,format="%Y-%m-%d")
        dataset.index = dataset['Date']
        dataset = dataset.sort_index(ascending=True, axis=0)
        dataset.fillna(0, inplace=True)
        dataset = dataset.loc[dataset['Stock']==stock_name]

    tf1.insert(END,str(filename))
    text.insert(END,"Dataset Loaded\n\n")
    text.insert(END,str(dataset.head()))

    plt.figure(figsize=(10,6), dpi=100)
    plt.plot(dataset.Date[0:20], dataset.Close[0:20], color='red')
    plt.title(stock_name+" Closing Price History")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.show()


def preprocessDataset():
    global dataset, sc
    global X_train, X_test, y_train, y_test

    text.delete('1.0', END)

    dataset = dataset.values
    Y = dataset[:,4:5]
    X = dataset[:,1:4]

    X = sc.fit_transform(X)
    Y = sc.fit_transform(Y)

    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

    text.insert(END,"Dataset Preprocessing Completed\n\n")
    text.insert(END,"Training size : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Testing size : "+str(X_test.shape[0])+"\n")


def calculateMSE(algorithm,predict,y_test):
    mse_value = mean_squared_error(y_test,predict)
    mse.append(mse_value)

    text.insert(END,algorithm+" MSE : "+str(mse_value)+"\n")
    text.insert(END,algorithm+" Accuracy : "+str(1-mse_value)+"\n\n")

    predict = sc.inverse_transform(predict).ravel()

    labels = y_test.reshape(y_test.shape[0],1)
    labels = sc.inverse_transform(labels).ravel()

    labels = labels[:100]
    predict = predict[:100]

    for i in range(20):
        text.insert(END,algorithm+" Predicted: "+str(predict[i])+"  Original: "+str(labels[i])+"\n")

    plt.plot(labels,color='red',label='Original')
    plt.plot(predict,color='green',label='Predicted')
    plt.title(algorithm+" Prediction")
    plt.legend()
    plt.show()


def runANN():
    text.delete('1.0', END)

    global mse
    global X_train, X_test, y_train, y_test
    mse=[]

    if os.path.exists('model/ann_model.json'):

        with open('model/ann_model.json',"r") as json_file:
            ann = model_from_json(json_file.read())

        ann.load_weights("model/ann_model_weights.h5")

    else:

        ann = Sequential()

        ann.add(Dense(50,activation='relu',input_shape=(X_train.shape[1],)))
        ann.add(Dense(50,activation='relu'))
        ann.add(Dense(1))

        ann.compile(optimizer='adam',loss='mean_squared_error')

        hist = ann.fit(X_train,y_train,epochs=1,batch_size=8,validation_data=(X_test,y_test))

        ann.save_weights('model/ann_model_weights.h5')

        with open("model/ann_model.json","w") as json_file:
            json_file.write(ann.to_json())

        pickle.dump(hist.history,open('model/ann_history.pckl','wb'))

    predict = ann.predict(X_test)

    calculateMSE("ANN",predict,y_test)


def runLSTM():

    text.delete('1.0', END)

    global X_train, X_test, y_train, y_test

    X_train1 = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
    X_test1 = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

    if os.path.exists('model/lstm_model.json'):

        with open('model/lstm_model.json',"r") as json_file:
            lstm = model_from_json(json_file.read())

        lstm.load_weights("model/lstm_model_weights.h5")

    else:

        lstm = Sequential()

        lstm.add(LSTM(50,return_sequences=True,input_shape=(X_train1.shape[1],1)))
        lstm.add(Dropout(0.2))

        lstm.add(LSTM(50,return_sequences=True))
        lstm.add(Dropout(0.2))

        lstm.add(LSTM(50))
        lstm.add(Dropout(0.2))

        lstm.add(Dense(1))

        lstm.compile(optimizer='adam',loss='mean_squared_error')

        hist = lstm.fit(X_train1,y_train,epochs=1,batch_size=8,validation_data=(X_test1,y_test))

        lstm.save_weights('model/lstm_model_weights.h5')

        with open("model/lstm_model.json","w") as json_file:
            json_file.write(lstm.to_json())

        pickle.dump(hist.history,open('model/lstm_history.pckl','wb'))

    predict = lstm.predict(X_test1)

    calculateMSE("LSTM",predict,y_test)


def graph():
    height = mse
    bars = ('ANN MSE','LSTM MSE')

    y_pos = np.arange(len(bars))

    plt.bar(y_pos,height)
    plt.xticks(y_pos,bars)
    plt.title("ANN vs LSTM MSE")
    plt.show()


def close():
    main.destroy()


font=('times',15,'bold')

title=Label(main,text='Stock Price Prediction using LSTM & ANN',bg='HotPink4',fg='yellow2',font=font)
title.place(x=0,y=5,width=1300,height=60)


font1=('times',13,'bold')

Label(main,text='Choose Dataset:',font=font1).place(x=430,y=100)

tf1=Entry(main,width=45,font=font1)
tf1.place(x=580,y=100)

Label(main,text='Choose Stock:',font=font1).place(x=50,y=100)

names=['AAPL','FB','MSFT','TSLA']

stock_list=ttk.Combobox(main,values=names,font=font1)
stock_list.place(x=210,y=100)
stock_list.current(0)

Button(main,text="Upload Dataset",command=uploadDataset,bg='#ffb3fe',font=font1).place(x=1020,y=100)

Button(main,text="Preprocess Dataset",command=preprocessDataset,bg='#ffb3fe',font=font1).place(x=50,y=150)

Button(main,text="Run ANN",command=runANN,bg='#ffb3fe',font=font1).place(x=300,y=150)

Button(main,text="Run LSTM",command=runLSTM,bg='#ffb3fe',font=font1).place(x=530,y=150)

Button(main,text="MSE Graph",command=graph,bg='#ffb3fe',font=font1).place(x=50,y=200)

Button(main,text="Exit",command=close,bg='#ffb3fe',font=font1).place(x=300,y=200)


text=Text(main,height=20,width=130,font=('times',13,'bold'))
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)

main.config(bg='plum2')

main.mainloop()