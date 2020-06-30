from tkinter import *
import numpy as np
import pandas as pd
import os
import librosa
import soundfile as sf
import argparse
import pyaudio
import numpy as np
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model, load_model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from scipy import signal
import time
import sys
import scipy, pylab

RATE     = int(44100)
CHUNK    = int(1024)
CHANNELS = int(1)
NUM_CHUNKS = 5
ind = 0
proc_ind = 2
len_window = 4096 #Specified length of analysis window
hop_length_ = 1024 #Specified percentage hop length between windows
xfade_in = np.linspace(0,1,num=CHUNK)
xfade_out = np.flip(xfade_in)

class Application(Frame):
    global make_sine2
    def make_sine2(seg_length,ii):
        global track1
        global track2 
        global CHUNK
        global encoder
        global full_net
        global full_net_graph
        global scales  
        global app 
        global num_latents
        global new_data
        global make_audio
        global alpha
        global len_window
        global tick 
        num_samps = seg_length*CHUNK
        make_audio = False
        num_samps = seg_length*CHUNK

        mysnip1 = track1[(num_samps*ii):(num_samps*(ii+1)+4*CHUNK)] #Snip Track 1
        mysnip2 = track2[(num_samps*ii):(num_samps*(ii+1)+4*CHUNK)] #Snip Track 2         

        _,_,A = signal.stft(mysnip1, nperseg=len_window, nfft=len_window, fs=44100, noverlap=3*1024) #STFT
        mag = A 
        mag = np.abs(mag) #Magnitude response of the STFT
        rememberA = mag.max(axis=0)+0.000000001 #Used for normalizing STFT frames (with addition to avoid division by zero)
        magA = mag / rememberA #Normalizing
        phaseA = np.angle(A) #Phase response of STFT
        magA = magA.T

        _,_,B = signal.stft(mysnip2, nperseg=len_window, nfft=len_window, fs=44100, noverlap=3*1024) #STFT
        mag = B 
        mag = np.abs(mag) #Magnitude response of the STFT
        rememberB = mag.max(axis=0)+0.000000001 #Used for normalizing STFT frames (with addition to avoid division by zero)
        magB = mag / rememberB #Normalizing
        phaseB = np.angle(B) #Phase response of STFT
        magB = magB.T

        temp_alpha = np.tile(alpha*scales,(NUM_CHUNKS+5,1)) #Match dims for XFade 1
        temp_negalpha = np.tile((1-alpha)*scales,(NUM_CHUNKS+5,1)) #Match dims for XFade 2
        temp_phase =alpha*phaseA+(1-alpha)*phaseB #Unstack and Interpolate Phase
        temp_remember = alpha*rememberA+(1-alpha)*rememberB #Unstack and Interpolate Normalizing gains
        temp_out_mag = full_net.predict([magA,magB,temp_alpha,temp_negalpha])

        out_mag = temp_out_mag.T * temp_remember
        E = out_mag*np.exp(1j*temp_phase)
        _, now_out = np.float32(signal.istft(0.24*E, fs=44100, noverlap=3*1024))
        out = now_out[CHUNK:-2*CHUNK]
        newdim = int(len(out)/CHUNK)
        print(len(out)/CHUNK)
        new_data = out.reshape((newdim,CHUNK))

    global callback 
    def callback(in_data, frame_count, time_info, status):
        global ind
        global proc_ind 
        global NUM_CHUNKS
        global all_data
        global new_data
        global make_audio
        global tick 
        global all_data

        if ind>=NUM_CHUNKS:
            ind = 0
            proc_ind+=1
        if ind==0:
            xfade = xfade_in*new_data[0,:] + xfade_out*all_data[-1,:]
            all_data = new_data
            all_data[0,:] = xfade
        if ind==1:
            make_audio = True
            tick = time.time()

        data = all_data[ind,:] #Send a chunk to the audio buffer when it asks for one
        ind +=1 
        return (data, pyaudio.paContinue)  

    def render(self):
        global mag1
        global phase1 
        global remember1
        global mag2
        global phase2 
        global remember2
        global CHUNK
        global encoder
        global enc_graph
        global decoder 
        global dec_graph
        global scales  
        global app 

        print(scales)
        temp_out_mag = mag1
        temp_phase = phase1
        temp_remember = remember1

        with enc_graph.as_default():
            temp_enc_mag = encoder.predict(temp_out_mag)
            enc_mag = temp_enc_mag * scales #NEED TO ADD SCALE HERE
        with dec_graph.as_default():
            temp_out_mag = decoder.predict(enc_mag)

        out_mag = temp_out_mag.T * temp_remember
        E = out_mag*np.exp(1j*temp_phase)
        out = np.float32(librosa.istft(E))
        out = (0.9/np.max(np.abs(out)))*out

        sf.write('rendered.wav', out, 44100, subtype='PCM_16')
        print('done rendering')


    def record(self):
        global mag
        global phase 
        global remember
        global CHUNK
        global encoder
        global enc_graph
        global decoder 
        global dec_graph
        global scales  
        global app 
        global recorded_scales
        global proc_ind

        first_ind = 0
        last_ind = 0
        total_frames = 0
        if self.RECORD_var.get() == 1:
            first_ind = proc_ind
            print('Button On')
            self.start_net()
        else:
            last_ind = proc_ind
            print('Button off')
            self.pause_sounds()

            total_frames = (last_ind-first_ind)*NUM_CHUNKS
            out_scales = np.ones((total_frames,15))
            temp_scales = np.vstack(recorded_scales)
            a = temp_scales.shape[0]
            increase_by = total_frames//a+1
            kurt=0
            for ii in range(a):
                the_rows = np.arange((kurt*increase_by),min(((kurt+1)*increase_by),total_frames))
                out_scales[the_rows,:] = np.tile(temp_scales[ii,:],(len(the_rows),1))
                kurt+=1
            ind_array = np.arange((first_ind),(NUM_CHUNKS*(last_ind)))
            temp_out_mag = mag[ind_array,:]
            temp_phase = phase[:,ind_array]
            temp_remember = remember[ind_array]

            with enc_graph.as_default():
                temp_enc_mag = encoder.predict(temp_out_mag)
                enc_mag = temp_enc_mag * out_scales #NEED TO ADD SCALE HERE
            with dec_graph.as_default():
                temp_out_mag = decoder.predict(enc_mag)

            out_mag = temp_out_mag.T * temp_remember
            E = out_mag*np.exp(1j*temp_phase)
            out = np.float32(librosa.istft(E))
            out = (0.9/np.max(np.abs(out)))*out

            sf.write('recorded.wav', out, 44100, subtype='PCM_16')
            print('done recording')



    def model_to_mem(self):
        global encoder
        global enc_graph
        global decoder 
        global dec_graph
        global full_net
        global full_net_graph

        data_path_net = os.path.join(os.getcwd(),'models/'+self.model_name.get()+'_trained_network.h5')
        full_net = load_model(data_path_net, compile=False)
        full_net._make_predict_function()
        full_net_graph = tf.get_default_graph()


    def process_track2(self):
        global mag1
        global phase1
        global remember1
        global mag2
        global phase2
        global remember2
        global track1
        global track2 

        len_window = 4096 #Specified length of analysis window
        hop_length_ = 1024 #Specified percentage hop length between windows

        filename_in = 'audio/'+self.track1_name.get()
        data_path = os.path.join(os.getcwd(),filename_in)
        track1, _ = librosa.load(data_path, sr=44100, mono=True)

        filename_in = 'audio/'+self.track2_name.get()
        data_path = os.path.join(os.getcwd(),filename_in)
        track2, _ = librosa.load(data_path, sr=44100, mono=True)

        print('tracks loaded')


    def start_net(self):
        global p 
        global stream
        global make_audio
        global NUM_CHUNKS
        global proc_ind
        global tick 

        tick = time.time()

        self.model_to_mem()

        make_audio = True
        make_sine2(NUM_CHUNKS,1)

        p = pyaudio.PyAudio()
        print("opening stream")
        stream = p.open(format=pyaudio.paFloat32,
                        channels=CHANNELS,
                        frames_per_buffer=CHUNK,
                        rate=RATE,
                        output=True,
                        stream_callback=callback)


        stream.start_stream()
        time.sleep(0.1)

    def pause_sounds(self):
        global p 
        global stream
        global ind
        global proc_ind 
        
        stream.stop_stream()
        print('sounds paused')
        stream.close()
        p.terminate()
        ind = NUM_CHUNKS+1
        proc_ind = 0

    def quit(self):
        root.destroy()
        

    def createWidgets(self):
        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit
        self.QUIT.pack()
        self.QUIT.place(relx=0.45,rely=0.95)

        self.model_name = Entry(self)
        self.model_name.pack()
        self.model_name.place(relx=0.4,rely=0.65)
        self.label = Label(self,text='Model Name')
        self.label.pack()
        self.label.place(relx=0.25,rely=0.65)

        self.track1_name = Entry(self)
        self.track1_name.pack()
        self.track1_name.place(relx=0.20,rely=0.6)
        self.label_1 = Label(self,text='Track 1')
        self.label_1.pack()
        self.label_1.place(relx=0.14,rely=0.6)

        self.track2_name = Entry(self)
        self.track2_name.pack()
        self.track2_name.place(relx=0.59,rely=0.6)
        self.label_2 = Label(self,text='Track 2')
        self.label_2.pack()
        self.label_2.place(relx=0.53,rely=0.6)

        self.START = Button(self)
        self.START["text"] = "START"
        self.START["fg"]   = "green"
        self.START["command"] =  lambda: self.start_net()
        self.START.pack()
        self.START.place(relx=0.45,rely=0.9)

        self.PAUSE = Button(self)
        self.PAUSE["text"] = "PAUSE"
        self.PAUSE["fg"]   = "black"
        self.PAUSE["command"] =  lambda: self.pause_sounds()
        self.PAUSE.pack()
        self.PAUSE.place(relx=0.45,rely=0.85)

        self.RECORD_var = IntVar()
        self.RECORD = Checkbutton(self, variable=self.RECORD_var)
        self.RECORD["text"] = "RECORD"
        self.RECORD["fg"]   = "black"
        self.RECORD["command"] =  lambda: self.record()
        self.RECORD.pack()
        self.RECORD.place(relx=0.45,rely=0.8)

        self.RENDER = Button(self)
        self.RENDER["text"] = "RENDER"
        self.RENDER["fg"]   = "black"
        self.RENDER["command"] =  lambda: self.render()
        self.RENDER.pack()
        self.RENDER.place(relx=0.45,rely=0.75)

        self.LOAD = Button(self)
        self.LOAD["text"] = "LOAD TRACKS"
        self.LOAD["fg"]   = "black"
        self.LOAD["command"] =  lambda: self.process_track2()
        self.LOAD.pack()
        self.LOAD.place(relx=0.45,rely=0.7)

        self.FADE = Scale(self,from_=100, to=0,length=300, orient='horizontal')
        self.FADE.set(50)
        self.FADE.pack()
        self.FADE.place(relx=0.30,rely=0.54)



    def createSliders(self):
        global scales 
        global num_latents
        scales = np.ones(num_latents)
        self.scale_list = []
        for w in range(num_latents):
            scale = Scale(self,from_=200, to=0,length=200)
            scale.pack()
            scale.place(relx=w/float(num_latents),rely=0.2)
            scale.set(100)
            scales[w]=scale.get()
            self.scale_list.append(scale)

    def update_scales(self):
        global scales 
        global recorded_scales
        global make_audio
        global NUM_CHUNKS
        global proc_ind
        global alpha
        global temp_scales

        alpha = self.FADE.get()/100.
        if make_audio:
            t = time.time()
            make_sine2(NUM_CHUNKS,proc_ind)


        for w in range(num_latents):
            temp_scales[w]=self.scale_list[w].get()/100.
        scales = temp_scales
        if self.RECORD_var.get() == 1:
            recorded_scales.append(scales)
        self.after(POLL_TIME, self.update_scales)


    def __init__(self, master=None):
        global recorded_scales
        global POLL_TIME
        global make_audio
        global alpha
        global temp_scales
        global all_data

        alpha = 1
        temp_scales = np.ones(num_latents)
        all_data = np.zeros((21,1024))

        make_audio = False
        POLL_TIME = 1

        Frame.__init__(self, master,width=800, height=800)
        self.pack()
        self.createWidgets()
        self.createSliders()
        recorded_scales = []
        self.update_scales()

global app 
global num_latents
num_latents = int(10)
root = Tk()
app = Application(master=root)
app.mainloop()
root.destroy()
