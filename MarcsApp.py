from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import webbrowser


forbiden = False
height= 300
width= 400
CameraNo = 0
PeopleCam = "videos\example_01.mp4"
limit = 10
mask_model = 'mask/model8.h5'
front_face_model = 'mask/haarcascade_frontalface_default.xml'
prototxtSSD = 'mobilenet_ssd\MobileNetSSD_deploy.prototxt'
caffemodelSSD = 'mobilenet_ssd\MobileNetSSD_deploy.caffemodel'

textInfo = """-------------------------------------------------------------\n
MARCS
Smart Door Lock Systems\n
-------------------------------------------------------------\n
Version ~ 1.0
Package Needed : Keras, Dlib, imutils, Opencv, numpy

Creator : >>Numero Uno Arroefy
>>Aretha Putri
>>Januar Excel
>>Evint Leovonzka

Credit : >>Prajna
>>Thakshila
>>Nouman Ahmad
"""

root = Tk()
root.title('Marcs')

def start():
    from mylib.centroidtracker import CentroidTracker
    from mylib.trackableobject import TrackableObject
    import imutils, dlib, cv2, time
    import numpy as np
    from keras.models import load_model
    

    val = None
    startbut['text'] = "Done"
    goset['text'] = 'Back to Main'
    startbut['command'] = stop
    goset['state'] = 'disabled'
    Ds = "DOOR STATUS : "

    

    model = load_model(mask_model)

    face_clsfr=cv2.CascadeClassifier(front_face_model)
    source=cv2.VideoCapture(CameraNo)
    source.set(3,300)
    source.set(4,300)

    labels_dict={0:'MASK',1:'NO MASK'}
    color_dict={0:(0,255,0),1:(0,0,255)}

    net = cv2.dnn.readNetFromCaffe(prototxtSSD,caffemodelSSD)

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

    vs = cv2.VideoCapture(PeopleCam)
    vs.set(3,300)
    vs.set(4,300)
    vs.set(5,30)


    W = None
    H = None
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    Ptotal = 0
    
    #holy loop
    try:
        while True:
            ret,img=source.read()
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=face_clsfr.detectMultiScale(img,1.5,3)
            _, frame = vs.read()
            frame = imutils.resize(frame,width=500)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            (H, W) = frame.shape[:2]

            status = "Waiting"
            rects = []

            if totalFrames % 30 == 0:
                status = "Detecting"
                trackers = []

                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                net.setInput(blob)
                detections = net.forward()

                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence > 0.4:
                        idx = int(detections[0, 0, i, 1])

                        if CLASSES[idx] != "person":
                            continue

                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")

                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)

                        trackers.append(tracker)

            else:
                for tracker in trackers:
                    status = "Tracking"

                    tracker.update(rgb)
                    pos = tracker.get_position()

                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    rects.append((startX, startY, endX, endY))
            
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
            cv2.putText(frame, "-Entrance Border-", (10, H - ((i * 20) + 200)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            objects = ct.update(rects)

            for (objectID, centroid) in objects.items():
                to = trackableObjects.get(objectID, None)

                if to is None:
                    to = TrackableObject(objectID, centroid)

                else:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    if not to.counted:
                        if direction < 0 and centroid[1] < H // 2:
                            totalUp += 1
                            to.counted = True

                        elif direction > 0 and centroid[1] > H // 2:
                            totalDown += 1
                            to.counted = True

                trackableObjects[objectID] = to

                Ptotal = int(int(totalDown)-int(totalUp))

                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

            for x,y,w,h in faces:
            
                face_img=gray[y:y+w,x:x+w]
                resized=cv2.resize(face_img,(100,100))
                normalized=resized/255.0
                reshaped=np.reshape(normalized,(1,100,100,1))
                result=model.predict(reshaped)

                label=np.argmax(result,axis=1)[0]

                area = w*h
                minarea = 10000
                maxarea = 50000

                if area > minarea and area<maxarea:
                    cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
                    cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
                    cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                
                if labels_dict[label] == 'MASK':
                    val = True

                elif labels_dict[label] == 'NO MASK':
                    val = False


            info = [
            ("Exit", totalUp),
            ("Enter", totalDown),
            ("Status", status),
            ]

            info2 = [
            ("Total people inside", Ptotal),
            ]

            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            for (i, (k, v)) in enumerate(info2):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


            if val == True and Ptotal <= limit:
                Ds = 'DOOR STATUS : Open'
                print(Ds)
                cv2.putText(img, Ds, (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                val = None

            elif val == True and Ptotal > limit:
                Ds = 'DOOR STATUS : Closed'
                print(Ds)
                cv2.putText(img, Ds, (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                val = None

            elif val == False and Ptotal > limit:
                Ds = 'DOOR STATUS : Closed'
                print(Ds)
                cv2.putText(img, Ds, (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                val = None
                
            elif val == False and Ptotal <= limit:
                Ds = 'DOOR STATUS : Closed'
                print(Ds)
                cv2.putText(img, Ds, (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                val = None

            elif val == None and Ptotal > limit:
                Ds = 'DOOR STATUS : Closed'
                print(Ds)
                cv2.putText(img, Ds, (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                val = None

            else:
                Ds = 'DOOR STATUS : Open'
                print(Ds)
                cv2.putText(img, Ds, (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


            cv2.imshow('Mask Detection',img)
            cv2.imshow("People Counter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("x"):
                break

            totalFrames += 1

        source.release()
        vs.release()
        cv2.destroyAllWindows()

    except ValueError:
        print('gagal')
     

def stop():
    doors['text'] = 'Smart Door Lock System'
    startbut['text'] = "Run"
    goset['text'] = 'Back to Main'
    startbut['command'] = start
    goset['state'] = 'normal'

def Run():
    global x, goset, startbut, doors
    y = 150
    x = 300
    canvas.delete('all')
    canvas.config(width=x,height=y)

    doors = Label(canvas,text="It take a while to run, so make sure your settings",bg='white', width=40, height=3)
    doors.after(1000,afterthis)
    startbut = Button(root, text="Run",width= 10, command=start)
    goset = Button(root, text="Settings", width= 10, command=Setting)
    lb = canvas.create_window(10,20,anchor=NW,window=doors)
    Rb = canvas.create_window(x/2 -80,100,anchor=NW,window=startbut)
    Sb = canvas.create_window(x/2+3, 100, anchor=NW, window=goset)

def afterthis():
    doors['text']="press 'x' to stop"
def BackMain():
    canvas.delete('all')
    canvas.config(width=width,height=height)

    lab =canvas.create_window(height/2 + 60,10,anchor=N, window=label)
    Rb = canvas.create_window(height/2 -20, width/2 - 120, anchor=NW, window=RunButton)
    Sb = canvas.create_window(height/2 - 20, width/2 - 90, anchor=NW, window=SetButton)
    Cb = canvas.create_window(height/2 - 20, width/2 - 60,anchor=NW, window=CloseButton)
    Ib = canvas.create_window(350, 280,anchor=W, window=InfoButton)

def BackSettings():
    canvas.delete('all')

    label = Label(canvas,text="Settings Menu",bg='white', width=40, height=2)
    label2 = Label(canvas,text="Note : All the settings will set to default after closing the program",bg='#CDEDF6')
    BackButton['command'] = BackMain
    BackButton['text'] = "Back to Main"
    
    Fm = canvas.create_window(height/2 - 20, width/2 - 120,anchor=NW, window=FaceMask)
    Pc = canvas.create_window(height/2 - 20, width/2 - 90,anchor=NW, window=PeCount)
    lab =canvas.create_window(height/2 + 60,10,anchor=N, window=label)
    lab2 =canvas.create_window(200,260, window=label2)
    Bb = canvas.create_window(height/2 - 20, width/2 - 60,anchor=NW, window=BackButton)

def Setting():
    canvas.delete("all")
    canvas.config(width=width,height=height)

    label = Label(canvas,text="Settings Menu",bg='white', width=40, height=2)
    label2 = Label(canvas,text="Note : All the settings will set to default after closing the program",bg='#CDEDF6')

    Fm = canvas.create_window(height/2 - 20, width/2 - 120,anchor=NW, window=FaceMask)
    Pc = canvas.create_window(height/2 - 20, width/2 - 90,anchor=NW, window=PeCount)
    lab =canvas.create_window(height/2 + 60,10,anchor=N, window=label)
    lab2 =canvas.create_window(200,260, window=label2)
    Bb = canvas.create_window(height/2 - 20, width/2 - 60,anchor=NW, window=BackButton)

def facemaskset():
    canvas.delete('all')
    global variable
    global note
    Option = [0,1,2,3]
    variable =IntVar()
    variable.set(CameraNo)
    select = ttk.Label(canvas,text="Select Camera(number) :",background='white')

    mb = ttk.OptionMenu(root,variable, Option[0],*Option,command=valcammask)

    label = Label(canvas,text="Face Mask Detection Settings",bg='white', width=40, height=2)
    note = ttk.Button(root,text="Important Note",command=Note)
    BackButton['command'] = BackSettings
    BackButton['text'] = "Back to Setting"

    lab =canvas.create_window(height/2 + 60,10,anchor=N, window=label)
    Bb = canvas.create_window(height/2 - 20, height -50,anchor=NW, window=BackButton)
    notebut = canvas.create_window(width/2 - 40 , 220,anchor=NW, window=note)
    setmenulab =canvas.create_window(height/4 + 30 , 100,anchor=NW, window=select)
    setmenu = canvas.create_window(height -40 , 100,anchor=NW, window=mb)

def Note():
    global Notes
    Notes = Toplevel(root)
    Notes.title('Important Notes')
    textInfo = """Camera/Webcam 0 is default camera for Mask Detection it's basically your default webcam
    Camera 1, 2, 3 it's just additional webcam, it's option for you to choose your webcam, but please make sure it exist\n
    For People counter you should use an overhead camera or CCTV, and if it's possible use wifi camera just put the ip address but if it's not possible
    I really sorry but you have to change the variable value by code(I will make a sign to the variable), but you can choose to test
    program with an overhead video\n
    limit default is 10 but you can set it to bigger number or smaller"""

    BackButton['state'] = 'disabled'
    note['state'] = 'disabled'

    infocanvas = Canvas(Notes,height= 160,width= 780, bg='#CDEDF6').pack()
    infolab = Label(Notes,text=textInfo, bg='#CDEDF6').place(relx=0)
    cb =Button(Notes, text="Close Info", command=CloseNote)
    cb.place(relx=0.45,rely=0.8)
    changeOnHoverButton(cb,"#4169E1","white")

    Notes.resizable(False,False)
    Notes.protocol("WM_DELETE_WINDOW", CloseNote)

def CloseNote():
    Notes.destroy()
    BackButton['state'] = 'normal'
    note['state'] = 'normal'

def valcammask(a):
    global CameraNo
    a = variable.get()
    CameraNo = a

def valLimit():
    global entry
    entry = Entry(root,width=15,textvariable=limit)
    eb = canvas.create_window(width/2 - 50,height/2-57,anchor=N, window=entry)
    SetRoomLimit['command'] = peoplecountset
    BackButton['state'] = 'disabled'
    Enter = Button(root,text="Enter Number",command=intval, width=12)
    Enb =canvas.create_window(width/2 + 50,height/2-60,anchor=N, window=Enter)

def intval():
    try:
        global limit
        val = int(entry.get())
        limit = val
        entry.config(textvariable=val)
        labelset['text'] = "People Counter Settings"
        BackButton['state'] = 'normal'
    except ValueError:
        labelset['text'] = "that's not number, please use number/int"

def cctv():
    global ipad
    ipad = Entry(root,width=25)
    ipad.insert(END, PeopleCam)
    open_file = ttk.Button(root,text='Select File',command=selectfile)
    CameraSet['command'] = peoplecountset
    labelset['text'] = "Insert Camera Ip address or select video files"
    ipd = canvas.create_window(width/2 -40,height - 147,anchor=N, window=ipad)
    openbutton = canvas.create_window(width-110,height - 150,anchor=N, window=open_file)
    Done = Button(root,text="Done",command=done, width=12)
    Enb =canvas.create_window(width/2 - 40,height - 122,anchor=NW, window=Done)

def done():
    global PeopleCam

    val = str(ipad.get())
    PeopleCam = val
    ipad.delete(0,END)
    ipad.insert(END, val)

    labelset['text'] = "People Counter Settings"

def selectfile():
    filetypes = (
        ('video files', '*.mp4'),
        ('All files', '*.*')
    )

    filename = filedialog.askopenfilename(
        title='Select a file',
        initialdir='Videos',
        filetypes=filetypes)
    ipad.delete(0,END)
    ipad.insert(END, filename)
    

def peoplecountset():
    canvas.delete('all')
    global note, labelset, SetRoomLimit,CameraSet

    labelset = Label(canvas,text="People Counter Settings",bg='white', width=40, height=2)
    SetRoomLimit = Button(root,text="Set Room Limit",width=30,command=valLimit)
    CameraSet = Button(root,text="Set Camera/Video",width=30,command=cctv)
    note = ttk.Button(root,text="Important Note",command=Note)

    BackButton['command'] = BackSettings
    BackButton['text'] = "Back to Setting"

    lab =canvas.create_window(height/2 + 60,10,anchor=N, window=labelset)
    Bb = canvas.create_window(height/2 - 20, height -50,anchor=NW, window=BackButton)
    Overhead = canvas.create_window(width/2 + 5, height/2 - 30,anchor=N, window=CameraSet)
    lim = canvas.create_window(width/2 + 5,height/2 -90,anchor=N, window=SetRoomLimit)
    notebut = canvas.create_window(width/2 - 40 , 220,anchor=NW, window=note)

def CloseInfo():
    top.destroy()

    RunButton['state'] = 'normal'
    SetButton['state'] = 'normal'
    CloseButton['state'] = 'normal'
    InfoButton['state'] = 'normal'

def Link(url):
    webbrowser.open_new(url)

def changeOnHover(text, colorOnHover, colorOnLeave):
    text.bind("<Enter>", func=lambda e: text.config(fg=colorOnHover))
    text.bind("<Leave>", func=lambda e: text.config(fg=colorOnLeave))

def changeOnHoverButton(button, colorOnHover, colorOnLeave):
    button.bind("<Enter>", func=lambda e: button.config(bg=colorOnHover))
    button.bind("<Leave>", func=lambda e: button.config(bg=colorOnLeave))

def Info():
    global top
    top = Toplevel()
    top.title('Program Information')

    infocanvas = Canvas(top,height= width-50,width= height + 5, bg='#CDEDF6').pack()
    infolab = Label(top,text=textInfo, bg='#CDEDF6').place(relx=0)
    infoGit = Label(top,text="My Github",bg='#CDEDF6')
    infoGit.place(relx=0.4, y=290)
    infoGit.bind("<Button-1>", lambda x: Link("https://github.com/UnoArroefy?tab=repositories"))
    changeOnHover(infoGit,"#4169E1","black")

    cb =Button(top, text="Close Info", command=CloseInfo)
    cb.place(relx=0.4,rely=0.9)
    changeOnHoverButton(cb,"#4169E1","white")

    RunButton['state'] = 'disabled'
    SetButton['state'] = 'disabled'
    CloseButton['state'] = 'disabled'
    InfoButton['state'] = 'disabled'

    top.resizable(False,False)
    top.protocol("WM_DELETE_WINDOW", CloseInfo)

canvas = Canvas(root,height= height,width= width, bg='#CDEDF6')
canvas.pack()

label = Label(canvas,text="Smart Door Lock System",bg='white', width=40, height=2)

RunButton = Button(root, text="Run Program",width= 20, command=Run)
SetButton = Button(root, text="Settings", width= 20, command=Setting)
CloseButton = Button(root,text="Exit", width=20 ,command=root.quit)
InfoButton = Button(root,text="Info", relief= FLAT, bg='#CDEDF6',activeforeground='#06AED5',activebackground='#CDEDF6',command=Info)

BackButton = Button(root,text="Back to Main", width=20, command=BackMain)
FaceMask = Button(root,text="Face Mask Detection", width=20, command=facemaskset)
PeCount = Button(root,text="People Counter", width=20, command=peoplecountset)

lab =canvas.create_window(height/2 + 60,10,anchor=N, window=label)

Rb = canvas.create_window(height/2 -20, width/2 - 120, anchor=NW, window=RunButton)
Sb = canvas.create_window(height/2 - 20, width/2 - 90, anchor=NW, window=SetButton)
Cb = canvas.create_window(height/2 - 20, width/2 - 60,anchor=NW, window=CloseButton)
Ib = canvas.create_window(350, 280,anchor=W, window=InfoButton)

changeOnHoverButton(RunButton,'#CDEDF6',"white")
changeOnHoverButton(SetButton,'#CDEDF6',"white")
changeOnHoverButton(CloseButton,'#CDEDF6',"white")
changeOnHover(InfoButton,"#4169E1","black")

root.resizable(False,False)
root.mainloop()