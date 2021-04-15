### 1. Notebook ###

运行结果： [T2-foodvoice-EDA.ipynb](https://github.com/frankyangdev/NLP-Learning/blob/main/FoodVoiceRecognition/T2-foodvoice-EDA.ipynb)

### 2. EDA Study ###

Python package Librosa 请参考task 1[笔记](https://github.com/frankyangdev/NLP-Learning/blob/main/FoodVoiceRecognition/Task1-Baseline-LearningNotes.md)

#### 2.1 [librosa.display.waveplot](https://librosa.org/doc/latest/generated/librosa.display.waveplot.html) ####

```
librosa.display.waveplot(y, sr=22050, max_points=50000.0, x_axis='time', offset=0.0, max_sr=1000, ax=None, **kwargs)
```

Plot the amplitude envelope of a waveform.

If y is monophonic, a filled curve is drawn between [-abs(y), abs(y)].

If y is stereo, the curve is drawn between [-abs(y[1]), abs(y[0])], so that the left and right channels are drawn above and below the axis, respectively.

Long signals (duration >= max_points) are down-sampled to at most max_sr before plotting.

* Plot a monophonic waveform (单音波形 )

```python
import matplotlib.pyplot as plt
y, sr = librosa.load(librosa.ex('choice'), duration=10)
fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
librosa.display.waveplot(y, sr=sr, ax=ax[0])
ax[0].set(title='Monophonic')
ax[0].label_outer()
```

* stereo waveform (立体声波形 )

```python

y, sr = librosa.load(librosa.ex('choice', hq=True), mono=False, duration=10)
librosa.display.waveplot(y, sr=sr, ax=ax[1])
ax[1].set(title='Stereo')
ax[1].label_outer()
```

* harmonic and percussive components with transparency (透明的谐波和打击乐成分 )

```python
y, sr = librosa.load(librosa.ex('choice'), duration=10)
y_harm, y_perc = librosa.effects.hpss(y)
librosa.display.waveplot(y_harm, sr=sr, alpha=0.25, ax=ax[2])
librosa.display.waveplot(y_perc, sr=sr, color='r', alpha=0.5, ax=ax[2])
ax[2].set(title='Harmonic + Percussive')
```

![image](https://user-images.githubusercontent.com/39177230/114903322-460d2580-9e49-11eb-9a67-4ce4e208e654.png)


#### 2.2 [librosa.display.specshow](https://librosa.org/doc/latest/generated/librosa.display.specshow.html) ####

```
librosa.display.specshow(data, x_coords=None, y_coords=None, x_axis=None, y_axis=None, sr=22050, hop_length=512, fmin=None, fmax=None, tuning=0.0, bins_per_octave=12, key='C:maj', Sa=None, mela=None, thaat=None, ax=None, **kwargs)
```

Display a spectrogram/chromagram/cqt/etc. (频谱图/色谱图 )

* Visualize an STFT power spectrum using default parameters

```python
import matplotlib.pyplot as plt
y, sr = librosa.load(librosa.ex('choice'), duration=15)
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
                               sr=sr, ax=ax[0])
ax[0].set(title='Linear-frequency power spectrogram')
ax[0].label_outer()
```

* on a logarithmic scale, and using a larger hop

```python
hop_length = 1024
D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)),
                            ref=np.max)
librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
                         x_axis='time', ax=ax[1])
ax[1].set(title='Log-frequency power spectrogram')
ax[1].label_outer()
fig.colorbar(img, ax=ax, format="%+2.f dB")
```
![image](https://user-images.githubusercontent.com/39177230/114903844-c895e500-9e49-11eb-9a86-ea5a10da14eb.png)





