import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import tkinter
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM,Dropout
from tensorflow.keras.utils import to_categorical

main = tkinter.Tk()
main.title("Intrusion Detection System")
main.geometry("1200x1000")

class prog:
	def upload():
		global filename
		text.delete('1.0',END)
		filename=askopenfilename(initialdir = "dataset")
		text.insert(END,"Dataset Loaded")
	def csv():
		global data
		global obj
		text.delete('1.0',END)
		data=pd.read_csv(filename)
		columns =(['real','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','real','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','label'])
		data.columns=columns
		text.insert(END,str(data.head()))
		#print(data.info())
		obj=data.copy()

		text.insert(END,"\n\nNull values= "+str(obj.isnull().values.sum())+"\n")
		text.insert(END,"\nprotocol_type:\n "+str(obj['protocol_type'].value_counts())+"\n") #to print diff types in that column 
		text.insert(END,"\nservice:\n "+str(obj['service'].value_counts())+"\n")
		text.insert(END,"\nflag:\n "+str(obj['flag'].value_counts())+"\n")
		text.insert(END,"\nlabel:\n "+str(obj['label'].value_counts())+"\n")

		labels4 = obj['label'].astype('category').cat.categories.tolist()
		replace_map_comp4 = {'label' : {k: v for k,v in zip(labels4,list(range(1,len(labels4)+1)))}}
		#print(replace_map_comp4)
		obj.replace(replace_map_comp4,inplace=True)

		labels1=obj['protocol_type'].astype('category').cat.categories.tolist()
		replace_map_comp1 = {'protocol_type' : {k: v for k,v in zip(labels1,list(range(1,len(labels1)+1)))}}
		#print(replace_map_comp1)
		obj.replace(replace_map_comp1,inplace=True)

		labels3=obj['flag'].astype('category').cat.categories.tolist()
		replace_map_comp3 = {'flag' : {k: v for k,v in zip(labels3,list(range(1,len(labels3)+1)))}}
		#print(replace_map_comp1)
		obj.replace(replace_map_comp3,inplace=True)

		labels2=obj['service'].astype('category').cat.categories.tolist()
		replace_map_comp2 = {'service' : {k: v for k,v in zip(labels2,list(range(1,len(labels2)+1)))}}
		#print(replace_map_comp1)
		obj.replace(replace_map_comp2,inplace=True)
		text.insert(END,"\n\nEncoded Dataset:\n"+str(obj.head()))
	def corre():
		correlation=obj.corr()
		text.delete('1.0',END)
		text.insert(END,str(correlation['label']))
	def divide():
		global X,y,X_train,X_test, y_train,y_test
		text.delete('1.0',END)
		f1=pd.read_csv(filename,usecols=[1,2,3,22,23,28,32,33,35,41])
		text.insert(END,"Dataset with selected features:\n"+str(f1.head()))
		columns=(['protocol_type','service','flag','count','srv_count','same_srv_rate','dst_host_srv_count','dst_host_same_srv_rate','dst_host_same_src_port_rate','label'])		
		f1.columns=columns
		labelencoder = LabelEncoder()
		f1['protocol_type'] = labelencoder.fit_transform(f1['protocol_type'])
		f1['service'] = labelencoder.fit_transform(f1['service'])
		f1['flag'] = labelencoder.fit_transform(f1['flag'])
		f1['label'] = labelencoder.fit_transform(f1['label'])
		text.insert(END,"\n\n\nEncoded Dataset:\n"+str(f1.head()))
		X = f1.iloc[:,:-1]
		y = f1.iloc[:,-1]
		X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
	def ann():
		model = Sequential()
		model.add(Dense(6, input_dim=9, activation='relu'))
		model.add(Dense(3, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(X_train, y_train, epochs=70, batch_size=10)
		predictions = model.predict_classes(X_test)
		print("\nThe label column of test dataset ",y_test)
		print("The predicted classes of test dataset ",predictions)

		accu = accuracy_score(y_test,predictions) * 100
		pr = precision_score(y_test,predictions,average='macro') * 100
		rec = recall_score(y_test,predictions,average='macro') * 100
		print("\nANN Prediction Accuracy : ",accu)
		print("\nANN Precision : ",pr)
		print("\nANN Recall : ",rec)
	def rnn():
		XX = X.values.reshape((X.shape[0], X.shape[1], 1))
		print(XX.shape)
		Y1 = to_categorical(y)
		X_train1, X_test1, y_train1, y_test1 = train_test_split(XX, Y1, test_size=0.2)
		lstm_model = Sequential()
		lstm_model.add(LSTM(128, input_shape=(X.shape[1],1), activation='relu', return_sequences=True))
		lstm_model.add(Dropout(0.2))
		lstm_model.add(LSTM(128, activation='relu'))
		lstm_model.add(Dropout(0.2))
		lstm_model.add(Dense(32, activation='relu'))
		lstm_model.add(Dropout(0.2))
		lstm_model.add(Dense(y_train1.shape[1], activation='softmax'))
		lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		acc_history = lstm_model.fit(XX, Y1, epochs=20, validation_data=(X_test1, y_test1))
		#print(lstm_model.summary())
		predict = lstm_model.predict(X_test1)
		predict = np.argmax(predict, axis=1)
		testY = np.argmax(y_test1, axis=1)
		print("\nThe label column of test dataset ",testY)
		print("The predicted classes of test dataset ",predict)
		#print(testY.shape)
		#print(predict.shape)

		acc = accuracy_score(testY,predict) * 100
		p = precision_score(testY,predict,average='macro') * 100
		r = recall_score(testY,predict,average='macro') * 100
		print("\nRNN Prediction Accuracy : ",acc)
		print("\nRNN Precision : ",p)
		print("\nRNN Recall : ",r)

font = ('times', 16, 'bold')
title = Label(main, text='Intrusion Detection System',anchor=CENTER)
title.config(bg='#DDD0CB', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)
title.place(x=0,y=7)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=prog.upload)
upload.place(x=900,y=100)
upload.config(font=font1)

df = Button(main, text="Reading Data ", command=prog.csv)
df.place(x=900,y=180)
df.config(font=font1)

df = Button(main, text="Correlation ", command=prog.corre)
df.place(x=900,y=250)
df.config(font=font1)

df = Button(main, text="Updated Dataset", command=prog.divide)
df.place(x=900,y=320)
df.config(font=font1)

df = Button(main, text="ANN ", command=prog.ann)
df.place(x=900,y=400)
df.config(font=font1)

df = Button(main, text="RNN ", command=prog.rnn)
df.place(x=900,y=480)
df.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=20,y=100)
text.config(font=font1)

main.config(bg='#F4EBE7')
main.mainloop()
