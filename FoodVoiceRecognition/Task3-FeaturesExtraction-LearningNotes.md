
### 1. Notebook ###

运行结果 [Task3-FeaturesExtraction.ipynb](https://github.com/frankyangdev/NLP-Learning/blob/main/FoodVoiceRecognition/Task3-FeaturesExtraction.ipynb)


### 2. Notebook 学习 ###

#### 2.1. MFCC特征提取知识 ####

* 过零率 （Zero Crossing Rate）
* 频谱质心 （Spectral Centroid）
* 声谱衰减 (Spectral Roll-off）
* 梅尔频率倒谱系数 （Mel-frequency cepstral coefficients ，MFCC）
* 色度频率 （Chroma Frequencies）

#### 2.2 在语音识别（SpeechRecognition）和话者识别（SpeakerRecognition）方面，最常用到的语音特征就是梅尔倒谱系数 （Mel-scaleFrequency Cepstral Coefficients，简称MFCC） ####

![image](https://user-images.githubusercontent.com/39177230/115114408-c35ba600-9fc1-11eb-97eb-50b711aaa27e.png)


根据人耳听觉机理的研究发现，人耳对不同频率的声波有不同的听觉敏感度。从200Hz到5000Hz的语音信号对语音的清晰度影响对大。两个响度不等的声音作用于人耳时，则响度较高的频率成分的存在会影响到对响度较低的频率成分的感受，使其变得不易察觉，这种现象称为掩蔽效应。由于频率较低的声音在内耳蜗基底膜上行波传递的距离大于频率较高的声音，故一般来说，低音容易掩蔽高音，而高音掩蔽低音较困难。在低频处的声音掩蔽的临界带宽较高频要小。所以，人们从低频到高频这一段频带内按临界带宽的大小由密到疏安排一组带通滤波器，对输入信号进行滤波。将每个带通滤波器输出的信号能量作为信号的基本特征，对此特征经过进一步处理后就可以作为语音的输入特征。

![image](https://user-images.githubusercontent.com/39177230/115111937-77a2ff80-9fb5-11eb-87df-ab8c21abb7d8.png)



* #### 快速傅立叶变换（Fast Fourier Transformation,FFT）：将时域信号变换成为信号的功率谱 ####

* #### 离散余弦变换（Discrete Cosine Transformation,DCT）：去除各维信号之间的相关性，将信号映射到低维空间 ####

* #### Windowing ####
Windowing involves the slicing of the audio waveform into sliding frames.
![image](https://user-images.githubusercontent.com/39177230/115111979-c781c680-9fb5-11eb-8ec0-aa79818374f7.png)

* #### Cepstrum — IDFT(inverse Discrete Fourier Transform 离散傅立叶变换) ####
 
Below is the model of how speech is produced.

![image](https://user-images.githubusercontent.com/39177230/115112065-33642f00-9fb6-11eb-8a28-dfc8762de31a.png)


Cepstrum is the reverse of the first 4 letters in the word “spectrum”. Our next step is to compute the Cepstral which separates the glottal source and the filter. Diagram (a) is the spectrum with the y-axis being the magnitude. Diagram (b) takes the log of the magnitude. Look closer, the wave fluctuates about 8 times between 1000 and 2000. Actually, it fluctuates about 8 times for every 1000 units. That is about 125 Hz — the source vibration of the vocal folds.

![image](https://user-images.githubusercontent.com/39177230/115112126-981f8980-9fb6-11eb-988c-b4b11c1a6be3.png)






### Reference ###

1. [语音特征参数MFCC提取过程详解](https://blog.csdn.net/jojozhangju/article/details/18678861)
2. [Speech Recognition — Feature Extraction MFCC & PLP](https://jonathan-hui.medium.com/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9)
3. [Wiki: Mel Scale](https://en.wikipedia.org/wiki/Mel_scale)
