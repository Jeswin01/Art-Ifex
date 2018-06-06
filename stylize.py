import tensorflow as tf
import numpy as np
import vgg16
import cv2
import math
import sys
from PyQt5 import QtWidgets,uic,QtCore,QtGui
from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon,QPixmap
import dummy1_rc
import traceback
import threading
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import time

import traceback
import threading
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from PyQt5.QtCore import *
import cv2
from PyQt5.QtCore import pyqtSlot
import subprocess
import os

qtCreatorFile="mainwindowui.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
style_image_path=""
content_video_path=""
global dstnvideoname
h=1
w=1
c=1
outlocation=""
class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    counter = pyqtSignal(int)
    counting = False
    w=1
    title ='Error'
    left = 900
    top = 400
    width = 320
    height = 200
    
    
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        pb = self.progressBar
        pb.hide()
        self.output_button.hide()
        self.image_select_button.clicked.connect(self.selectImageFile)
        self.video_select_button.clicked.connect(self.selectVideoFile)
        self.output_button.clicked.connect(self.output_button_function)
        self.Output_location_button.clicked.connect(self.select_output_location)
        
        self.counter.connect(pb.setValue) 
        self.submit_button.clicked.connect(self.getInput)
        #self.submit_button.clicked.connect(self.something)
        #self.submit_button.clicked.connect(self.getInput)
        #time.sleep(2)
        #self.threadpool = QThreadPool()
        #print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
    def executnfn(self):
        #style_filename = 'style7.jpg'
        style_filename = style_image_path
        style_image = self.load_image(style_filename)
        content_layer_ids = [4]
        style_layer_ids = [1,2,3,4]
        content_video_path
        global outlocation
        #cap = cv2.VideoCapture("rabbit.mp4")
        cap = cv2.VideoCapture(content_video_path)
        count=0
        #dstnvideoname='output3.avi'
        out = cv2.VideoWriter(outlocation,-1, 20.0, (self.w,self.h))
        model = vgg16.VGG16()
        session = tf.InteractiveSession(graph=model.graph)
        img=np.array([1,1,1])
        self.submit_button.hide()
        pb=self.progressBar
        
        loss_style = self.create_style_loss(session=session,
                                   model=model,
                                   style_image=style_image,
                                   layer_ids=style_layer_ids)
        loss_denoise = self.create_denoise_loss(model)
        
        
        
        while(count<=211):
            ret, frame = cap.read()
            count = count + 1
            
            if(count>100) and count%10==0:
                x=int((count-100)*100/111)
                #x=count-100
                #print("ch"+str(x))
                self.counter.emit(x)
                print(x)
                QApplication.processEvents()
                
                img=self.style_transfer(content_image=frame,
                                        loss_denoise=loss_denoise,
                                        loss_style=loss_style,
                                        prvsframe=img,
                                        session=session,
                                        model=model,
                                        style_image=style_image,
                                        content_layer_ids=content_layer_ids,
                                        style_layer_ids=style_layer_ids,
                                        weight_content=content_weight_value,
                                        weight_style=style_weight_value,
                                        weight_denoise=99,
                                        weight_temporal=1111,
                                        num_iterations=20,
                                        step_size=8)
                
                out.write(img)
                print(count)
        pb.hide()
        session.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    def hello(self):
        
        iti=iti+1
        print("inside iti")
        
        
        #self.hey()
        #for iti in 100:
           # print(iti)
           
    def startCounting(self):
        if not self.counting:
            self.counting = True
            thread = threading.Thread(target=self.something)
            thread.start()

    def something(self):
        pb = self.progressBar
        pb.show()
        for x in range(100):
                self.counter.emit(x)
                time.sleep(0.1)
        self.counting = False
        
    def hey(self):
        print("inside hey")
        global iti
        iti=0
        #print("d1")
        pb = self.progressBar
        pb.show()
        #print("d2")
        #self.connect(self.get_thread, SIGNAL("setValue(QString)"), self.setValue)
        #self.connect(self.get_thread, SIGNAL("finished()"), self.done)
        #print("d3")
        #self.get_thread.start()
        #worker = Worker()
        #self.threadpool.start(worker)
        #for j in range(10):
        #self.hello()
        print ("bye")
         
    def selectImageFile(self):
        #print("d1")
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'e:\\',"Image files (*.jpg *.gif)")
        #print(fname[0])
        global style_image_path
        style_image_path=fname[0]
        #print("d2")
        l1=self.label1
        #print("d13")
        if(style_image_path==""):
            self.initUI("Image not selected")

        if(style_image_path!=""):
            pixmap = QPixmap(style_image_path)
            #print("d14")
            #print(pixmap)
            #print(b1.width(),b1.height())
            l1.setPixmap(pixmap)
            #b1.resize(pixmap.width(),pixmap.height())
            
            #print("d2")
 
    def initUI(self,msg):
        super().__init__()
        #print("chkin")
        
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        buttonReply = QMessageBox.critical(self, self.title,msg,QMessageBox.Retry,QMessageBox.Retry)
        if buttonReply == QMessageBox.Retry:
            return
 
        show()
 
        


    def selectVideoFile(self):
        #print("d1")
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'e:\\',"Media files (*.mkv *.avi *.mp4)")
        print(fname[0])
        global content_video_path
        content_video_path=fname[0]
        #print("d2")
        l2=self.label2
        if(content_video_path==""):
            self.initUI("Video not selected")
        
        if(content_video_path!=""):
            cap = cv2.VideoCapture(content_video_path)
            print("added 13-3-18")
            i=1
            no_of_frames = int(cap.get(7))
            #print(cap.set(0,10000*60))
            fps=int(cap.get(5))
            if(no_of_frames<=10000):
                cap.set(0,1)
                ret, frame1 = cap.read()
            elif(no_of_frames>86400):
                cap.set(0,5*1000*60)
                
            else:
                cap.set(0,10000)
                
            ret, frame1 = cap.read() 
            #print(frame1)
            #cv2.imshow('wname',frame1)
            cv2.imwrite('new1.jpg',frame1)
            self.h,self.w,self.c=frame1.shape
            pixmap=QPixmap('new1.jpg')
            l2.setPixmap(pixmap)
            cap.release()
            #return content_video_path
            
    def select_output_location(self):
        fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', "","All Files (*);;Text Files (*.txt);;Media files (*.mkv *.avi *.mp4)")
        print(fname)
        if fname[0]=="":
            self.initUI("Output File path not selected")
        else:    
            #fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Save file', 'c:\\',"Media files (*.mkv *.avi *.mp4)")
            fname=fname[0]
            dot_position=fname.index('.')
            fname=fname[:dot_position]
            print(fname)
            global outlocation
            outlocation=fname+".avi"
            l3=self.label3
            l3.setText(outlocation)
            #outlocation[0]=outlocation[0]+"\\"
            print(outlocation)
            
    def getInput(self):
        global style_weight_value
        global outlocation,content_video_path,style_image_path
        style_weight_value=(self.style_weight_spinner.value())
        print(style_weight_value)
        global content_weight_value
        content_weight_value=(self.content_weight_spinner.value())
        print(content_weight_value)

        enter_loop=1
        #print("chkpont1")
        if(int(style_weight_value)==0):
            self.initUI("Style weight is not set")
            enter_loop=0
            
        #print("chkpont2")
        if(int(content_weight_value)==0):
            self.initUI("Content weight is not set")
            enter_loop=0
            
        #print("chkpont3")    
        if(style_image_path==""):
            self.initUI("Image not selected")
            enter_loop=0
            
        #print("chkpont4")    
        if(content_video_path==""):
            self.initUI("Video not selected")
            enter_loop=0

        if(outlocation==""):
            self.initUI("Output File path not selected")
            enter_loop=0
            
            
        if(enter_loop==1):
            pb = self.progressBar
            pb.show()
            self.counter.emit(1)
            self.executnfn()
            #print("hai")
           # for i in range(0, 100):
            #    time.sleep(1)#correct time
            #    time.sleep(0.005)# for debugging ,faster
              #  pb.setValue(i)
                #QApplication.processEvents()

            #pb.close()
            self.submit_button.hide()
            
            self.output_button.show()
            #self.groupBox1l.hide()
            #self.groupBox2r.hide()

            
            
            #price = int(self.price_box.toPlainText())
            #tax = (self.tax_rate.value())
            ##total_price_string = "The total price with tax is: " + str(total_price)
            #self.results_window.setText(total_price_string)
            #self.setWindowTitle("Video Player")
            #self.setGeometry(500,400,400,300)
            
            #self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
     
            #videoWidget = QVideoWidget()
            #self.mediaPlayer.setVolume(100)
            
            #self.openButton = QPushButton()
            #self.playButton = QPushButton()
            #self.playButton.setEnabled(False)
            #print(content_video_path)
            semipath='E:/How_to_do_style_transfer_in_tensorflow-master/'
            dstnvideoname='output3.avi'
            global dstnpath
            dstnpath=semipath + dstnvideoname
            '''content_video_path=dstnpath'''
            #self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(content_video_path)))
            #self.playButton.setEnabled(True)
            #self.openButton.setEnabled(True)
            #self.openButton.clicked.connect(self.openFile)
            #self.playButton.clicked.connect(self.play)
            #self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            
            
     
            #self.positionSlider = QSlider(Qt.Horizontal)
            #self.positionSlider.setRange(0, 0)
            #self.positionSlider.sliderMoved.connect(self.setPosition)
     
            #self.errorLabel = QLabel()
            #self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
            #        QSizePolicy.Maximum)
     
            # Create new action
            #openAction = QAction(QIcon('open.png'), '&Open', self)        
            #openAction.setShortcut('Ctrl+O')
            #openAction.setStatusTip('Open movie')
            #openAction.triggered.connect(self.openFile)
     
            # Create exit action
            #exitAction = QAction(QIcon('exit.png'), '&Exit', self)        
            #exitAction.setShortcut('Ctrl+Q')
            #exitAction.setStatusTip('Exit application')
            #exitAction.triggered.connect(self.exitCall)
     
            # Create menu bar and add action
            #menuBar = self.menuBar()
            #fileMenu = menuBar.addMenu('&File')
            #fileMenu.addAction(newAction)
            #fileMenu.addAction(openAction)
            #fileMenu.addAction(exitAction)
            
            # Create a widget for window contents
            #wid = QWidget(self)
            #self.setCentralWidget(wid)
                
            # Create layouts to place inside widget
            #controlLayout = QHBoxLayout()
            #controlLayout.setContentsMargins(0, 0, 0, 0)
            #controlLayout.addWidget(self.openButton)
            #controlLayout.addWidget(self.playButton)
            #controlLayout.addWidget(self.positionSlider)
     
            #layout = QVBoxLayout()
            #layout.addWidget(videoWidget)
            #layout.addLayout(controlLayout)
            #layout.addWidget(self.errorLabel)
            #self.errorLabel.hide()
            # Set widget to contain window contents
            #wid.setLayout(layout)
            #self.openFile()
           
     
            #self.mediaPlayer.setVideoOutput(videoWidget)
            #self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
            #self.mediaPlayer.positionChanged.connect(self.positionChanged)
            #self.mediaPlayer.durationChanged.connect(self.durationChanged)
            #self.mediaPlayer.error.connect(self.handleError)def output_button_function(self):
    def output_button_function(self):
            global outlocation
            global style_image_path
            global content_video_path
            outlocation=outlocation.replace("/","\\")
            print(outlocation)
            p = subprocess.Popen([os.path.join("D:\\","Program Files (x86)","VideoLAN","VLC","vlc.exe"),outlocation])
            time.sleep(2)
            self.output_button.hide()
            self.submit_button.show()
            style_image_path=""
            content_video_path=""
            outlocation=""
            #sys.exit(app.exec_())
            pixmap = QPixmap("E:\\How_to_do_style_transfer_in_tensorflow-master\\blueprism.jpg")
            print(pixmap)
            #print(pixmap)
            #print(b1.width(),b1.height())
            l1=self.label1
            l2=self.label2
            l3=self.label3
            l1.setPixmap(pixmap)
            l2.setPixmap(pixmap)
            l1.setText("No image selected")
            l2.setText("No image selected")
            l3.setText("Output File path not selected")
            #self.textEdit.setText("")
            self.style_weight_spinner.setValue(0.00)
            self.content_weight_spinner.setValue(0.00)
            pb = self.progressBar
            pb.hide()
                
 
    def openFile(self):
        #fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
         #       QDir.homePath())
        #print(fileName)
        content_video_path=dstnpath 
        #print("passed var:"+content_video_path)
        fileName=content_video_path
        if fileName!= '':
            #print(fileName)
            #self.mediaPlayer.setMedia(
             #       QMediaContent(QUrl.fromLocalFile(fileName)))
            #self.playButton.setEnabled(True)
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(content_video_path)))
            self.playButton.setEnabled(True)
            self.playButton.clicked.connect(self.play)
 
    def exitCall(self):
        sys.exit(app.exec_())
        
    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()
 
    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))
 
    def positionChanged(self, position):
        self.positionSlider.setValue(position)
 
    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)
 
    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)
 
    def handleError(self):
        #self.playButton.setEnabled(False)
        self.errorLabel.setText("Error Info: " + self.mediaPlayer.errorString())
 
 
#if __name__ == '__main__':
#    app = QApplication(sys.argv)
#    player = VideoWindow()
#    player.resize(640, 480)
#    player.show()
#    sys.exit(app.exec_())
        
    @pyqtSlot(int)   
    def setValue(self, val): # Sets value
        #print("val in setValue:"+str(val))
        self.progressBar.setProperty("value", int(val))

#class Worker(QRunnable):
    #@pyqtSlot()
#    def run(self):
#        print("T start")
#        print(iti)
#        print("T end")

#class progressUpdater(QThread):
#    def __init__(self):
#        QThread.__init__(self)
#        self.progress = progress#my latest creation

  #  def __del__(self):
  #      self.wait()
        
   # def run(self):
    #    j=0
     #   print(str(j)+"hey")
     #   
      #  #for subreddit in self.subreddits:
       # #    top_post = self._get_top_post(subreddit)
       # for j in range(100):
       #     self.emit(SIGNAL('setValue(QString)'),str(j))
       #     self.sleep(2)
        
    def load_image(self,filename, max_size=224):
        image = cv2.imread(filename)

        if max_size != 224:
        
            factor = max_size / np.max(image.size)
    
        # Scale the image's height and width.
            size = np.array(image.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
            size = size.astype(int)

        # Resize the image.
            image = image.resize(size, 1)

    # Convert to numpy floating-point array.
        return np.float32(image)
        
    def create_content_loss(self,session, model, content_image, layer_ids):    
        feed_dict = model.create_feed_dict(image=content_image)
        layers = model.get_layer_tensors(layer_ids)
        values = session.run(layers, feed_dict=feed_dict)
        with model.graph.as_default():        
            layer_losses =[]            
            for value, layer in zip(values, layers):
                # These are the values that are calculated
                # for this layer in the model when inputting
                # the content-image\
                value_const = tf.constant(value)
                loss = self.mean_squared_error(layer, value_const)
                # Add the loss-function for this layer to the
                # list of loss-functions.
                layer_losses.append(loss)
        
            total_loss = tf.reduce_mean(layer_losses)
        return total_loss
        
    def mean_squared_error(self,a, b):
        return tf.reduce_mean(tf.square(a - b))
        
    def gram_matrix(self,tensor):
        #gram matrix is vector of dot products for vectors
        #of the feature activations of a style layer
    
        #4d tensor from convolutional layer
        shape = tensor.get_shape()
    
        # Get the number of feature channels for the input tensor,
        # which is assumed to be from a convolutional layer with 4-dim.
        num_channels = int(shape[3])

        #-1 means whatever number makes the data fit 
        # Reshape the tensor so it is a 2-dim matrix. This essentially
        # flattens the contents of each feature-channel.
        matrix = tf.reshape(tensor, shape=[-1, num_channels])
    
    
        #gram matrix is transpose matrix with itself
        #so each entry in gram matrix
        #tells us if a feature channel has a tendency
        #to be activated with another feature channel
    
        #idea is to make the mixed image match patterns from style image
    
    
        # Calculate the Gram-matrix as the matrix-product of
        # the 2-dim matrix with itself. This calculates the
        # dot-products of all combinations of the feature-channels.
        gram = tf.matmul(tf.transpose(matrix), matrix)

        return gram 
        
    def create_style_loss(self,session, model, style_image, layer_ids):


        # Create a feed-dict with the style-image.
        feed_dict = model.create_feed_dict(image=style_image)


        layers = model.get_layer_tensors(layer_ids)

        with model.graph.as_default():

            gram_layers = [self.gram_matrix(layer) for layer in layers]


            values = session.run(gram_layers, feed_dict=feed_dict)

            layer_losses = []
    

            for value, gram_layer in zip(values, gram_layers):
            
                value_const = tf.constant(value)

            
                loss = self.mean_squared_error(gram_layer, value_const)

                layer_losses.append(loss)

            total_loss = tf.reduce_mean(layer_losses)
        return total_loss       
        
    def create_denoise_loss(self,model):
        loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
               tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))

        return loss
    
    def create_temp_loss(self,prvs1,next1):

        if(prvs1.all()!=0) :
            return 1
        else :
            next1=next1.astype(np.uint8)
            prvs = cv2.cvtColor(prvs1,cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(prvs1)
            hsv[...,1] = 255
            next = cv2.cvtColor(next1,cv2.COLOR_BGR2GRAY)
            temporal_loss=0

            flow = cv2.calcOpticalFlowFarneback( prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            hsvback = np.zeros_like(next1)
            hsvback[...,1] = 255
            flowback = cv2.calcOpticalFlowFarneback( next,prvs, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magback, angback = cv2.cartToPolar(flowback[...,0], flowback[...,1])
            hsvback[...,0] = angback*180/np.pi/2
            hsvback[...,2] = cv2.normalize(magback,None,0, 255,cv2.NORM_MINMAX)
            bgrback = cv2.cvtColor(hsvback,cv2. COLOR_HSV2BGR)
            x=bgrback
            forward_warped_flow=bgr+ bgrback
            stylized_frame=bgrback
            D=len(x)*len(x[0])*len(x[0][0] )
            res=( (np.abs(forward_warped_flow + bgrback) )**2)-0.01*(np.abs(forward_warped_flow)**2+np.abs(bgrback)**2)-0.5
            res=np.ceil(res)
            res2=np.clip(res,0,1)
            c=np.logical_not(res)
            c=c.astype(int)
            newres=c*((next1-bgr) **2)
            newressum=newres.sum()
            temporal_loss=(1/D)*newressum*10**3# temporal_loss=(1/D)*k=1Dsigma ck(xk-wk)2
            #print(temporal_loss)
            xc=tf.constant(temporal_loss,dtype=np.float32)
            return xc
        
    def style_transfer(self,content_image,loss_denoise,loss_style,prvsframe,session,model, style_image,
                   content_layer_ids, style_layer_ids,
                   weight_content=1.5, weight_style=10.0,weight_denoise =0.3,weight_temporal=1111,num_iterations=120, step_size=10.0):
   
        
        
        # Create a TensorFlow-session.
        '''session = tf.InteractiveSession(graph=model.graph)'''



        # Create the loss-function for the content-layers and -image.
        loss_content = self.create_content_loss(session=session,
                                       model=model,
                                       content_image=content_image,
                                       layer_ids=content_layer_ids)

        # Create the loss-function for the style-layers and -image.
            

        # Create the loss-function for the denoising of the mixed-image.
        
        
       
        mixed_image = content_image.astype(np.float32)
        
        #loss_temporal = tf.Variable(1e-10, name='loss_temporal')
  
        
        

        # Initialize the adjustment values for the loss-functions.
        adj_content = tf.Variable(1e-10, name='adj_content')
        adj_style = tf.Variable(1e-10, name='adj_style')
        adj_denoise = tf.Variable(1e-10, name='adj_denoise')

        session.run([adj_content.initializer,
                 adj_style.initializer,
                 adj_denoise.initializer])
   
        update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
        update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
        update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))
        #loss_temporal = loss_temporal.assign(self.create_temp_loss(prvsframe,mixed_image))
        
        
        
        loss_combined = weight_content * adj_content * loss_content + \
                        weight_style * adj_style * loss_style + \
                        weight_denoise * adj_denoise * loss_denoise 
                        #weight_temporal * loss_temporal
        #update_adj_temporal = adj_temporal.assign(1.0 / (self.create_temp_loss(prvsframe,mixed_image) + 1e-10))
        
        gradient = tf.gradients(loss_combined, model.input)    
        # List of tensors that we will run in each optimization iteration.-
        run_list = [gradient, update_adj_content, update_adj_style, update_adj_denoise]

        # The mixed-image is initialized with random noise.
        # It is the same size as the content-image.
        #where we first init it
        

        for i in range(num_iterations):
            # Create a feed-dict with the mixed-image.
            feed_dict = model.create_feed_dict(image=mixed_image)

            
            '''if(prvsframe.any()):
                update_adj_temporal=adj_temporal.assign(1.0 / 1e-8 + self.create_temp_loss(prvsframe,mixed_image))
            else :
                update_adj_temporal=adj_temporal.assign(0)'''
        
            grad, adj_content_val, adj_style_val ,adj_denoise_val= session.run(run_list, feed_dict=feed_dict)
            
            #print(adj_content_val, adj_style_val ,adj_denoise_val,temp_val)
            grad = np.squeeze(grad)

     
            step_size_scaled = step_size / (np.std(grad) + 1e-8)

            # Update the image by following the gradient.
            #gradient descent
            mixed_image -= grad * step_size_scaled

        
            mixed_image = np.clip(mixed_image, 0.0, 255.0)
            
                

            # Print a little progress-indicator.
            print(". ", end="")

            # Display status once every 10 iterations, and the last.
            if (i % 10 == 0) or (i == num_iterations - 1):
                print()
                #print("Iteration:", i)
            
        print()
        #print("Final image:")
        mixed_image=self.plot_image_big(mixed_image)

        # Close the TensorFlow session to release its resources.
        '''session.close()'''
    
        # Return the mixed-image.
        return mixed_image

    def plot_image_big(self,image):
        # Ensure the pixel-values are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)

        # Convert pixels to bytes.
        
        image = image.astype(np.uint8)
        return image
        # Convert to a PIL-image and display it.
        
#content_filename = 'G:/style/images/elon_musk.jpg'

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())





#content_image = load_image(content_filename)

