from tkinter import *
import cv2
from tkinter import messagebox
from tkinter import filedialog

root = Tk()
root.title("AI Detecting Objek")
root.iconbitmap('AI_1.ico')
root.geometry('925x500+300+200')
root.resizable(False, False)

bg = PhotoImage(file="AI_BGR.png")
my_label = Label(root, image=bg)
my_label.place(x=0 , y=0, relwidth=1, relheight=1)

def tampilanawal():
    Judul =  Label(root, text='AI Detecting Objek', bg='#001e33', fg='yellow',
                      font=('Comic Sans MS', 30, 'bold'))
    Judul.place(x=260, y=50)
    def DataObjekIng():
        file_data = open('objek.names', 'r')
        frame1 = Frame(root, width=925, height=500, bg='#b83dba')
        frame1.place(x=10, y=70)
        labeld = Label(root, text='Daftar Deteksi Objek', bg='#001e33', fg='yellow',
                      font=('Comic Sans MS', 30, 'bold'))
        labeld.place(x=5, y=5)

        def kembali():
            frame1.destroy()
            labeld.destroy()
            back.destroy()
            list_objek()

        back = Button(root, width=20, height=2, text='Kembali', border=0, bg='#57a1f8', cursor='hand2',
                          fg='white', command=kembali)
        back.place(x=150, y=440)

        text_box = Text(frame1, height=22, width=50, wrap='word')
        text_box.insert('end', file_data.read())
        text_box.pack(side=LEFT, expand=True)

        sb = Scrollbar(frame1)
        sb.pack(side=RIGHT, fill=BOTH)

        text_box.config(yscrollcommand=sb.set)
        sb.config(command=text_box.yview)

    def DataObjekIndo():
        file_data = open('Barang.names', 'r')
        frame1 = Frame(root, width=925, height=500, bg='#b83dba')
        frame1.place(x=10, y=70)
        labeld = Label(root, text='Daftar Deteksi Objek', bg='#001e33', fg='yellow',
                      font=('Comic Sans MS', 30, 'bold'))
        labeld.place(x=5, y=5)

        def kembali():
            frame1.destroy()
            labeld.destroy()
            back.destroy()
            list_objek()

        back = Button(root, width=20, height=2, text='Kembali', border=0, bg='#57a1f8', cursor='hand2',
                          fg='white', command=kembali)
        back.place(x=150, y=440)

        text_box = Text(frame1, height=22, width=50, wrap='word')
        text_box.insert('end', file_data.read())
        text_box.pack(side=LEFT, expand=True)

        sb = Scrollbar(frame1)
        sb.pack(side=RIGHT, fill=BOTH)

        text_box.config(yscrollcommand=sb.set)
        sb.config(command=text_box.yview)

    def kamera():
        Judul.destroy()
        bahasa.destroy()
        utama.destroy()
        keluar.destroy()
        label_1 = Label(root, text='Deteksi Objek', bg='#001e33', fg='yellow',
                        font=('Comic Sans MS', 30, 'bold'))
        label_1.place(x=60, y=50)
        labelind = Label(root, text='Bahasa Indonesia', bg='#001e33', fg='yellow',
                        font=('Comic Sans MS', 20, 'bold'))
        labelind.place(x=85, y=140)
        button1 = Button(root, width=19, pady=7, text='Kamera Internal', bg='#57a1f8', fg='white', cursor='hand2',
                         border=0, command=lambda:DetectObjek(0,0))
        button1.place(x=60, y=200)
        button2 = Button(root, width=19, pady=7, text='Kamera External', bg='#57a1f8', fg='white',
                         cursor='hand2',
                         border=0, command=lambda:DetectObjek(1,0))
        button2.place(x=210, y=200)
        labeing = Label(root, text='Bahasa Inggris', bg='#001e33', fg='yellow',
                        font=('Comic Sans MS', 20, 'bold'))
        labeing.place(x=100, y=270)
        button3 = Button(root, width=19, pady=7, text='Kamera Internal', bg='#57a1f8', fg='white', cursor='hand2',
                         border=0, command=lambda: DetectObjek(0,1))
        button3.place(x=60, y=330)
        button4 = Button(root, width=19, pady=7, text='Kamera External', bg='#57a1f8', fg='white',
                         cursor='hand2',
                         border=0, command=lambda: DetectObjek(1,1))
        button4.place(x=210, y=330)

        def kembali():
            label_1.destroy()
            labelind.destroy()
            labeing.destroy()
            button1.destroy()
            button2.destroy()
            button3.destroy()
            button4.destroy()
            button5.destroy()
            tampilanawal()

        button5 = Button(root, width=29, pady=7, text='kembali', bg='#57a1f8', fg='white',
                         cursor='hand2',
                         border=0, command=kembali)
        button5.place(x=100, y=400)


    def DetectObjek(ch,ch2):
        messagebox.showinfo("attention", "Tekan q berulang kali untuk berhenti merekam")
        thres = 0.5  # ambang batas untuk mendeteksi objek
        cap = cv2.VideoCapture(ch)  # mengambil camera internal (0), external(1)
        cap.set(3,640)
        cap.set(4,720)

        if cap is None or not cap.isOpened():
            messagebox.showinfo("WARNING","Kamera tidak tersedia")
        if ch2 == 0:
            classFile = 'Barang.names'
        else:
            classFile = 'objek.names'
        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightPath = 'frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightPath, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        while True:
            success, img = cap.read()
            classIds, confs, bbox = net.detect(img, confThreshold=thres)
            print(classIds, bbox)
            if not success: # untuk memberitahu user jika device tidak dapat menerima frame
                messagebox.showinfo("WARNING","Tidak dapat menerima frame")
                break
            if len(classIds) != 0:  # mencegah eror apabila tidak ada barang yang ke detect
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)  # membuat kotak

                    cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                                # margin text objek (dalam kotak)
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 2)  # Font,Ukuran,warna,ketebalan

                    cv2.putText(img, (str(round(confidence * 100, 2)) + "%"), (box[0] + 200, box[1] + 30),
                                # margin text akurasi
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 2)  # Font,Ukuran,warna,ketebalan

            cv2.imshow("AI Detecting Objek", img)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def list_objek():
        Judul.destroy()
        bahasa.destroy()
        utama.destroy()
        keluar.destroy()

        def Indonesia():
            button1.destroy()
            button2.destroy()
            button3.destroy()
            labelk.destroy()
            label_1.destroy()
            back.destroy()
            DataObjekIndo()

        def Inggris():
            button1.destroy()
            button2.destroy()
            button3.destroy()
            labelk.destroy()
            label_1.destroy()
            back.destroy()
            DataObjekIng()

        label_1 = Label(root, text='Daftar Deteksi Objek', bg='#001e33', fg='yellow',
                      font=('Comic Sans MS', 30, 'bold'))
        label_1.place(x=60, y=50)

        def keawal():
            button1.destroy()
            button2.destroy()
            button3.destroy()
            labelk.destroy()
            label_1.destroy()
            back.destroy()
            tampilanawal()

        def open_txt():
            button1.destroy()
            button2.destroy()
            button3.destroy()
            labelk.destroy()
            label_1.destroy()
            back.destroy()
            text_file = filedialog.askopenfilename(initialdir="C:/gui/" , title="Open File Text", filetypes=(("Text Files", "*.names"), ))
            text_file1 = open(text_file, 'r')
            stuff = text_file1.read()

            my_text = Text(root,width=40, height=17, font=("Helvetica", 16))
            my_text.pack(pady=10)

            my_text.insert(END, stuff)
            text_file1.close()

            open_button = Button(root, text="Open Text File",command=open_txt)
            open_button.pack(pady=15)

            def kembali():
                my_text.destroy()
                open_button.destroy()
                balik.destroy()
                list_objek()

            balik = Button(root, width=20, height=2, text='Kembali', border=0, bg='#57a1f8', cursor='hand2',
                          fg='white', command=kembali)
            balik.place(x=385, y=440)

        button1 = Button(root, width=39, pady=7, text='Open File Text', bg='#57a1f8', fg='white', cursor='hand2',
               border=0,command=open_txt)
        button1.place(x=120, y=140)
        button2 = Button(root, width=39, pady=7, text='Objek Bahasa Indonesia', bg='#57a1f8', fg='white', cursor='hand2',
               border=0, command=Indonesia)
        button2.place(x=120, y=220)
        button3 = Button(root, width=39, pady=7, text='Objek Bahasa Inggris', bg='#57a1f8', fg='white', cursor='hand2',
               border=0,command=Inggris)
        button3.place(x=120, y=300)
        labelk=Label(root,text='Kembali Ke Tampilan Awal',fg='white',bg='red',font=('Microsoft Yahei UI Light',9,'bold'))
        labelk.place(x=150,y=360)
        back=Button(root,width=6,text='[ Back ]',border=0,bg='red',cursor='hand2',fg='blue',command=keawal)
        back.place(x=333,y=360)

    bahasa = Button(root, width=39, pady=7, text='Daftar List Objek Detecting', bg='#57a1f8', fg='white', cursor='hand2',
           border=0, command=list_objek)
    bahasa.place(x=320, y=160)
    utama = Button(root, width=39, pady=7, text='Detecting Objek', bg='#57a1f8', fg='white', cursor='hand2', border=0,
           command=kamera)
    utama.place(x=320, y=240)
    keluar = Button(root, width=39, pady=7, text='Keluar Program', bg='#57a1f8', fg='white', cursor='hand2', border=0,
                   command=root.quit)
    keluar.place(x=320, y=320)

tampilanawal()
root.mainloop()