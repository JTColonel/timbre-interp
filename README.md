# timbre-interp
Autoencoder Based Real-Time Timbre Interpolation Algorithm 

Example of interpolating between two timbres, no morphing: 
https://youtu.be/qu0Qozt_JNg

Example of interpolating between the same two timbres, but with morphing:
https://youtu.be/95sTJ9Whc7A

Example with harmonic conent and drum loops:
https://youtu.be/o_t3JXFlDvg

Example with two drum loops: 
https://youtu.be/7_VxFqUT_HE 


Tested with Python 3.6.6 

This application requires Ffmpeg and Port Audio 

```
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt-get install ffmpeg libav-tools
```

I suggest using a virtualenvironment to ensure that all packages are correct

```
mkdir venv
python -m venv venv/
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r 3req.txt
```

In order to generate a corpus to train your autoencoder, run wav2frames.py on the wav file of your choice. Please ensure the audio file is placed in the ```audio``` directory.
```
python wav2frames.py --filename_in=my_audio.wav --filename_out=my_corpus
```

Then, to train an autoencoder, run 
```
python train-network.py --filename_in=my_corpus --filename_out=my_embedding --n_epochs=100
```

To start the timbre interpolation program, run 

```
python timbre-interp.py
```

Type the name of the tracks you would like to filter into the "Track" boxes. Be sure these audio files are placed in the ```audio``` directory.

Type the prefix of the trained model you would like to run (in this case just ```my_embedding```) into the "Model Name" box.

Clicking "START" will start to filter the track through the neural network and play out audio in real time. Change the value of the sliders to change the latent representation of the audio. 

Clicking "PAUSE" will pause the audio output and freeze the track where it is. I'm pretty sure clicking "START" again will resume the track.

Clicking "QUIT" will close the application.




