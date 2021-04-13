### 1. Notebook: 
CNN Baseline运行结果: [T1-CNN-Baseline.ipynb](https://github.com/frankyangdev/NLP-Learning/blob/main/FoodVoiceRecognition/T1-CNN-Baseline.ipynb)

![image](https://user-images.githubusercontent.com/39177230/114564501-f260d780-9ca2-11eb-8185-5f4eef6b5776.png)


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
![image](https://user-images.githubusercontent.com/39177230/114537382-b74fab80-9c84-11eb-8940-626ea901f530.png)

不同层之间的实例化参数不全相同，具体参考文档[keras 常用层](https://keras-cn.readthedocs.io/en/latest/layers/core_layer)

**Dense层**
```python
keras.layers.core.Dense(
    units, 
    activation=None, 
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', 
    kernel_regularizer=None, 
    bias_regularizer=None, 
    activity_regularizer=None,
    kernel_constraint=None, 
    bias_constraint=None)
```

重要参数如下：

* units：大于0的整数，代表该层的输出维度。
* activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
* use_bias: 布尔值，是否使用偏置项

Code example:

```python
model.add(Dense(500,input_shape=(784,))) # 输入层，28*28=784  
model.add(Activation('tanh')) # 激活函数是tanh  
model.add(Dropout(0.5)) # 采用50%的dropout

model.add(Dense(500)) # 隐藏层节点500个  
model.add(Activation('tanh'))  
model.add(Dropout(0.5))

model.add(Dense(10)) # 输出结果是10个类别，所以维度是10  
model.add(Activation('softmax')) # 最后一层用softmax作为激活函数

```
![image](https://user-images.githubusercontent.com/39177230/114538069-84f27e00-9c85-11eb-8f3e-976150fd3e1d.png)


**3. 编译**

```python
compile(self, 
        optimizer, 
        loss, 
        metrics=[], 
        loss_weights=**None**,
        sample_weight_mode=**None**)
```

* optimizer：优化器，为预定义优化器名或优化器对象，参考优化器
* loss：目标函数，为预定义损失函数名或一个目标函数，参考目标函数
* metrics：列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=['accuracy']如果要在多输出模型中为不同的输出指定不同的指标，可像该参数传递一个字典，例如metrics={'ouput_a': 'accuracy'}
* sample_weight_mode：如果你需要按时间步为样本赋权（2D权矩阵），将该值设为“temporal”。默认为“None”，代表按样本赋权（1D权）。如果模型有多个输出，可以向该参数传入指定sample_weight_mode的字典或列表。在下面fit函数的解释中有相关的参考内容。
* kwargs：使用TensorFlow作为后端请忽略该参数，若使用Theano作为后端，kwargs的值将会传递给 K.function

```python
# 第三步：编译
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # 优化函数，设定学习率（lr）等参数  
model.compile(loss='categorical_crossentropy', optimizer=sgd) # 使用交叉熵作为loss函数

```

**4.训练**
```python
fit(self, x, y, 
    batch_size=32,
    nb_epoch=10, 
    verbose=1, 
    callbacks=[], 
    validation_split=0.0, 
    validation_data=None, 
    shuffle=True, 
    class_weight=None, 
    sample_weight=None)

```

本函数将模型训练nb_epoch轮，其参数有：

* x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
* y：标签，numpy array
* batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
* nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为"number of"的意思
* verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
* callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数
* validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。
* validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
* shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。
* class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）
* sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，* sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。

fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况
```python
(X_train, y_train), (X_test, y_test) = mnist.load_data() # 使用Keras自带的mnist工具读取数据（第一次需要联网）
# 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维  
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]) 
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])  
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int) 
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

model.fit( X_train, Y_train,batch_size=200,
          epochs=50,shuffle=True, verbose=1, validation_split=0.3)
model.evaluate(X_test, Y_test, batch_size=200, verbose=0)


```

**5. 预测 ***

```python
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)

```

本函数按batch计算在某些输入数据上模型的误差，其参数有：

* x：输入数据，与fit一样，是numpy array或numpy array的list
* y：标签，numpy array
* batch_size：整数，含义同fit的同名参数
* verbose：含义同fit的同名参数，但只能取0或1
* sample_weight：numpy array，含义同fit的同名参数
```python
predict(self, x, batch_size=32, verbose=0)
```

* 本函数按batch获得输入数据对应的输出，其参数有：

* 函数的返回值是预测值的numpy array

```python
'''
    第五步：输出
'''
print("test set")
scores = model.evaluate(X_test,Y_test,batch_size=200,verbose=0)
print("The test loss is %f" % scores)
result = model.predict(X_test,batch_size=200,verbose=0)

result_max = numpy.argmax(result, axis = 1)
test_max = numpy.argmax(Y_test, axis = 1)

result_bool = numpy.equal(result_max, test_max)
true_num = numpy.sum(result_bool)
print("The accuracy of the model is %f" % (true_num/len(result_bool)))

```

#### 2.5 CNN的基本知识

1、卷积神经网络(Convolutional Neural Networks，CNN)的作用：
    * cnn跟全连接的区别：原来一个输出神经元的计算是跟所有输入层的神经元相连，现在只是局部输入层的
                        神经元相连；同一所个通道中所有神经元的计算共享同一组权重和偏置
 
    CNN可以有效的降低 传统深度神经网络 参数量与计算量。
    * FC DDN 参数量与连接数：
        参数量 = 连接数 = 输入featureMap总特征数（尺寸） * 输出featureMap总特征数（尺寸）
 
    * CNN参数量(卷积核):  跟模型总的训练时间和次数有很大关系，参数量越多花的训练时间越长，越难训练；
                      参数量就是W数(卷积核尺寸，例：3*3*3*2)
      CNN链接数量(卷积)： 跟当次训练的时间有关，越多计算量越大，训练时间越长；
                      连接数的数量就是参数量（即卷积核的大小）*输出层的尺寸(卷积核大小)*输出层的尺寸。 
                      注意b加的位置就是每一个输出通道所有输入通道计算结果总和 + b
 
      CNN下采样（池化）的连接数： filter尺寸(4)*通道数*输出层的尺寸
      CNN全连接的连接量： 等于参数量，等于输入层尺寸(2维)*输入层通道数*输出尺寸(1维)
      
2、常见的CNN结构有：
    LeNet-5、AlexNet、ZFNet、VGGNet、GoogleNet、ResNet等等，其中在LVSVRC2015冠军ResNet是AlexNet的20多倍，
    是VGGNet的8倍；
    其中：AlexNet、ZFNet、GoogleNet 是过度的神经网络
         * LeNet-5：结构最为简单，常用作小型简单的项目，例如嵌入式芯片常常用到LeNet-5
         * VGGNet16、ResNet（151层）：模型复杂，适合处理大型的项目，占用资源较高。
         
3、从这些结构来讲CNN发展的一个方向就是层次的增加，通过这种方式可以利用增加的非线性得出目标函数的近似结构，同时得出更好
    的特征表达，但是这种方式导致了网络整体复杂性的增加，使网络更加难以优化，很容易过拟合。
    
4、CNN的应用主要是在图像分类和物品识别等应用场景应用比较多



**Reference**：

1. [keras 快速入门](https://blog.csdn.net/qq_40791129/article/details/113925142)
2. [Keras中文文档](https://keras-cn.readthedocs.io/en/latest/layers/core_layer/)
3. [CNN（卷积神经网络）](https://blog.csdn.net/qq_16555103/article/details/89914946)
