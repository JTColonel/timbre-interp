# timbre-interp
Autoencoder Based Real-Time Timbre Interpolation Algorithm 

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
pip install -r 3req.txt
```

To start the program, run 

```
python timbre-interp.py
```

Type the relative path of the track you would like to filter into the "Track Name" box.

Type the prefix of the trained model you would like to run (in this case just ```all_frames```) into the "Model Name" box.

Clicking "START" will start to filter the track through the neural network and play out audio in real time. Change the value of the sliders to change the latent representation of the audio. 

Clicking "PAUSE" will pause the audio output and freeze the track where it is. I'm pretty sure clicking "START" again will resume the track.

To render an entire track with fixed latent activations, click "RENDER". The song will be output as "rendered.wav" in your given directory. It should be a mono wav file, 16bit PCM, 44.1kHz.

To begin a recording of you altering the latents as the track plays, click "RECORD" and begin moving the sliders. 
To end a recording, just click the "RECORD" button again so that it is unchecked. The recorded wav file will be output as "recorded.wav" in your given directory. It should be a mono wav file, 16bit PCM, 44.1kHz.

Clicking "QUIT" will close the application.


