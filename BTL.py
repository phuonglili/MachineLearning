import tkinter as tk #GUI
from tkinter import *
#Thêm các thư viện để xử lí 
import tkinter as tk
from tkinter import messagebox
import numpy as np  #numpy để tính toán giá trị cho ma trận và mảng
import matplotlib.pyplot as plt #matplotlib để vẽ các sơ đồ cho tập dữ liệu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.sparse import coo_matrix 
from sklearn.metrics import accuracy_score #Tính xác suất sai số
import scipy.io as sio


#Khai báo các thư viện dùng để tạo giao diện cho ứng dụng ở đây tôi dùng Tkinter
#Thiết lập các giá trị mặc định cho giao diện như title hay kích thước của form hiện ra
window=tk.Tk()
window.title("Hồi quy logistic dự đoán nguy cơ bị đột quỵ")
window.geometry("800x600")
window.resizable(0, 0)
window.configure(background="#FFFFFF")


#Khai báo các biến thông tin nhập giá trị input
Name=StringVar()
Sex=IntVar()
Age=IntVar()
Hypertension=IntVar()
Heart=IntVar()
Glu_level=DoubleVar()
Bmi=DoubleVar()
Smokes=IntVar()

labelHead=Label(window,text="Nhập thông tin dưới đây",fg="#000000",font=("Arial  ",20) )
labelHead.grid(column=1,row=0,pady=20)

labelX=Label(window,text="Họ tên",fg="#000000",font=("Arial  ",17) )
labelX.grid(column=0,row=2,padx=10)
txt = Entry(window,textvariable = Name,width=30,font=("Arial",15))
txt.grid(column=1,row=2,padx=10,pady=15)

#####################Tuổi###################

labelX=Label(window,text="Tuổi",fg="#000000",font=("Arial  ",17) )
labelX.grid(column=0,row=3,padx=10)
Ages = Entry(window,textvariable = Age,width=30,font=("Arial",15))
Ages.grid(column=1,row=3,padx=10,pady=15)

#####################Giới tính###################
labelX=Label(window,text="Giới tính",fg="#000000",font=("Arial  ",17) )
labelX.grid(column=0,row=5,padx=10)

rad1 = Radiobutton(window, text="Nam",font=("Arial  ",15) ,variable=Sex, value=0)
rad1.grid(column=1, row=5)


rad2 = Radiobutton(window, text="Nữ",font=("Arial  ",15) ,variable=Sex, value=1)
rad2.grid(column=2, row=5)

#############################Tình trạng cao huyết áp#################
labelX=Label(window,text="Có bị cao huyết áp không?",fg="#000000",font=("Arial  ",17) )
labelX.grid(column=0,row=7,padx=10)

rad3 = Radiobutton(window, text="Có",font=("Arial  ",15) ,variable=Hypertension, value=1)
rad3.grid(column=1, row=7)


rad4 = Radiobutton(window, text="Không",font=("Arial  ",15) ,variable=Hypertension, value=0)
rad4.grid(column=2, row=7)
##########################Tình trạng bênh tim#####################

labelX=Label(window,text="Có bị bệnh tim không?",fg="#000000",font=("Arial  ",17))
labelX.grid(column=0,row=9,padx=10)

rad5 = Radiobutton(window, text="Có",font=("Arial  ",15) ,variable=Heart, value=1)
rad5.grid(column=1, row=9)


rad6 = Radiobutton(window, text="Không",font=("Arial  ",15) ,variable=Heart, value=0)
rad6.grid(column=2, row=9)


labelX=Label(window,text="Lượng đường trong máu :",fg="#000000",font=("Arial  ",17) )
labelX.grid(column=0,row=18,padx=10)
Glu_levels = Entry(window,textvariable = Glu_level,width=30,font=("Arial",15)).grid(column=1,row=18,padx=10,pady=15)

##########################SHORT OF BREATH#####################

labelX=Label(window,text="Chỉ số BMI:",fg="#000000",font=("Arial  ",17) )
labelX.grid(column=0,row=20,padx=10)
Bmis = Entry(window,textvariable = Bmi,width=30,font=("Arial",15)).grid(column=1,row=20,padx=10,pady=15)

##########################Smokes#####################

labelX=Label(window,text="Hút thuốc",fg="#000000",font=("Arial  ",17) )
labelX.grid(column=0,row=22,padx=10)

rad9 = Radiobutton(window, text="Có",font=("Arial  ",15) ,variable=Smokes, value=1)
rad9.grid(column=1, row=22)


rad10 = Radiobutton(window, text="Không",font=("Arial  ",15) ,variable=Smokes, value=0)
rad10.grid(column=2, row=22)


#Truyền data vào 2 biến từ 2 tập dữ liệu với input là Train_x.txt và nhãn cho tập dữ liệu trong Train_y.txt
file_x='Train_x.txt'
data=pd.read_csv(file_x,sep='\t')
#Chuyển data vừa đưa vào về dạng numpy.array
X = data.values
file_y="Train_y.txt"
data=pd.read_csv(file_y,sep='\t')
y=data.values
y=y.reshape(299)
print(y )

file_x='Test_x.txt'
data=pd.read_csv(file_x,sep='\t')
#Chuyển data vừa đưa vào về dạng numpy.array
test_x = data.values

file_y="Test_y.txt"
data=pd.read_csv(file_y,sep='\t')
test_y=data.values
test_y.reshape(29)


#Sử dụng thư viện trong sklearn để training cho tập ví dụ huấn luyện
pd= LogisticRegression(solver='lbfgs', max_iter=1000) 
pd.fit(X,y)

def getvalue():
    Name=txt.get()
# Hàm predict để đưa dữ liệu vào 
def predict():
   Sepb =  float(Sex.get())
   Ages =  float(Age.get()) 
   Hypertensions =  float(Hypertension.get())
   Hearts =  float(Heart.get())  
   Glu_levels =  float(Glu_level.get()) 
   Bmis =  float(Bmi.get()) 
   Smokess = float(Smokes.get())
   #Hàm pd.predict để đưa ra nhãn cho tập dữ liệu,giá trị trả về sẽ là 0 hoặc 1
   i2= pd.predict([[Sepb,Ages,Hypertensions,Hearts, Glu_levels,Bmis,Smokess]])
   probability= pd.predict_proba([[Sepb,Ages,Hypertensions,Hearts,Glu_levels,Bmis,Smokess]])
   pb=probability[0][1]
   pb=round(pb,4)
   i=i2
   if Ages ==0 or Glu_levels == 0 or Bmis == 0 :
        messagebox.showwarning("Chưa điền đủ dữ liệu", "Mời bạn nhập đủ dữ liệu")
   else:
        if i>=1:
                i=1   
                messagebox.showinfo( "Dự đoán ", " Bạn có %.2f %% nguy cơ bị đột quỵ! \n Vui lòng đến bệnh viện để kiểm tra sức khỏe!"%(pb*100))

        else:
                i=0
                messagebox.showinfo( "Dự đoán"," Chúc mừng!Bạn không có nguy cơ bị đột quỵ")
   

tk.Button(window,  text='Dự Đoán',borderwidth=5,font= ('Arial', 15, 'underline'),foreground='#ecf0f1',background="#0000FF",  command=predict).grid(column=1,row=24)
   

window.mainloop()

from sklearn.metrics import accuracy_score
from scipy import misc
clf4= LogisticRegression(solver='lbfgs', max_iter=1000) 
clf4.fit(X,y)

y_pred = clf4.predict(test_x)
print ("Độ chính xác thuật toán : %.2f %%" %(100*accuracy_score(test_y, y_pred)))
