### 1. Notebook: 
CNN Baseline运行结果: [T1-CNN-Baseline.ipynb](https://github.com/frankyangdev/NLP-Learning/blob/main/FoodVoiceRecognition/T1-CNN-Baseline.ipynb)

### 2. Notebook Study:

#### 2.1 unzip -qq

-q perform operations quietly (-qq = even quieter). Ordinarily unzip prints the names of the files it's extracting or testing, the extrac‐ tion methods, any file or zipfile comments that may be stored in the archive, and possibly a summary when finished with each archive. The -qq options suppress the printing of some or all of these messages.

#### 2.2 Python Library: [librosa](https://librosa.org/doc/latest/index.html) 



librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

**Overview**


The **librosa** package is structured as collection of submodules:

  - librosa

    - :ref:`librosa.beat <beat>`
        Functions for estimating tempo and detecting beat events.

    - :ref:`librosa.core <core>`
        Core functionality includes functions to load audio from disk, compute various
        spectrogram representations, and a variety of commonly used tools for
        music analysis.  For convenience, all functionality in this submodule is
        directly accessible from the top-level `librosa.*` namespace.
        
    - :ref:`librosa.decompose <decompose>`
        Functions for harmonic-percussive source separation (HPSS) and generic
        spectrogram decomposition using matrix decomposition methods implemented in
        *scikit-learn*.

    - :ref:`librosa.display <display>`
        Visualization and display routines using `matplotlib`.  

    - :ref:`librosa.effects <effects>`
        Time-domain audio processing, such as pitch shifting and time stretching.
        This submodule also provides time-domain wrappers for the `decompose`
        submodule.

    - :ref:`librosa.feature <feature>`
        Feature extraction and manipulation.  This includes low-level feature
        extraction, such as chromagrams, Mel spectrogram, MFCC, and various other
        spectral and rhythmic features.  Also provided are feature manipulation
        methods, such as delta features and memory embedding.

    - :ref:`librosa.filters <filters>`
        Filter-bank generation (chroma, pseudo-CQT, CQT, etc.).  These are primarily
        internal functions used by other parts of *librosa*.

    - :ref:`librosa.onset <onset>`
        Onset detection and onset strength computation.

    - :ref:`librosa.segment <segment>`
        Functions useful for structural segmentation, such as recurrence matrix
        construction, time-lag representation, and sequentially constrained
        clustering.

    - :ref:`librosa.sequence <sequence>`
        Functions for sequential modeling.  Various forms of Viterbi decoding,
        and helper functions for constructing transition matrices.

    - :ref:`librosa.util <util>`
        Helper utilities (normalization, padding, centering, etc.)


#### 2.3 Python Package: [tqdm](https://pypi.org/project/tqdm/)

**tqdm** derives from the Arabic word taqaddum (تقدّم) which can mean “progress,” and is an abbreviation for “I love you so much” in Spanish (te quiero demasiado).

Instantly make your loops show a smart progress meter - just wrap any iterable with tqdm(iterable), and you’re done!

```python
from tqdm import tqdm
for i in tqdm(range(100000)):
    print(i)
```

![image](https://user-images.githubusercontent.com/39177230/114500238-37ace700-9c5a-11eb-9ba7-817b86297f15.png)

#### 2.4 Python Deep Learning API [keras](https://keras.io/getting_started/)

Keras是一个高层神经网络API，Keras由纯Python编写而成并基 Tensorflow、 Theano以及 CNTK后端。Keras 为支持快速实验而生，能够把你的idea迅速转换为结果，如果你有如下需求，请选择Keras：
* 简易和快速的原型设计（keras具有高度模块化，极简，和可扩充特性）
* 支持CNN和RNN，或二者的结合
* 无缝CPU和GPU切换


keras编程能够划分为五个步骤：

1. 选择模型
2. 构建神经网络结构
3. 编译
4. 训练
5. 预测

**1. 选择模型**
Keras有两种类型的模型，顺序模型（[Sequential](https://keras-cn.readthedocs.io/en/latest/legacy/models/sequential/)）和泛型模型（[Model](https://keras-cn.readthedocs.io/en/latest/legacy/models/model/)），本文选择的是简单的顺序模型。直接实例化对象即可：

```python
model = Sequential()
```

**2. 构建网络**

神经网络中的数据层包括全连接层(Dense)、激活层(Activation)、随机失活层（Dropout）、Flatten层、Reshape层、Permute层、RepeatVector层、Lambda层、ActivityRegularizer层、Masking层。每一层的功能如下：





**Reference**：

1. [keras 快速入门](https://blog.csdn.net/qq_40791129/article/details/113925142)