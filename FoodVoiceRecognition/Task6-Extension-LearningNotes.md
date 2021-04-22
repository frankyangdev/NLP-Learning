### 1. HMM ###

![image](https://user-images.githubusercontent.com/39177230/115689830-08e7ec80-a38f-11eb-91f5-40ec0db39100.png)


**HMM的主要应用方向**

单分子动力学分析、密码分析、语音识别、语音合成、词性标记、扫描解决方案中的文档分离、机器翻译、局部放电、基因预测生物序列比对、时间序列分析、活动识别蛋白质折叠、变态病毒检测、DNA主题发现

**NLP方向：**

* 词性标注：给定一个词的序列（也就是句子），找出最可能的词性序列（标签是词性）。如ansj分词和ICTCLAS分词等。
* 分词：给定一个字的序列，找出最可能的标签序列（断句符号：[词尾]或[非词尾]构成的序列）。结巴分词目前就是利用BMES标签来分词的，B（开头）,M（中间),E(结尾),S(独立成词）
* 命名实体识别：给定一个词的序列，找出最可能的标签序列（内外符号：[内]表示词属于命名实体，[外]表示不属于）。如ICTCLAS实现的人名识别、翻译人名识别、地名识别都是用同一个Tagger实现的。
机器翻译

### 2. GMM ###

高斯密度函数估计是一种参数化模型。高斯混合模型（Gaussian Mixture Model, GMM）是单一高斯概率密度函数的延伸，GMM能够平滑地近似任意形状的密度分布。高斯混合模型种类有单高斯模型（Single Gaussian Model, SGM）和高斯混合模型（Gaussian Mixture Model, GMM）两类。类似于聚类，根据高斯概率密度函数（Probability Density Function, PDF）参数不同，每一个高斯模型可以看作一种类别，输入一个样本x，即可通过PDF计算其值，然后通过一个阈值来判断该样本是否属于高斯模型。很明显，SGM适合于仅有两类别问题的划分，而GMM由于具有多个模型，划分更为精细，适用于多类别的划分，可以应用于复杂对象建模。

**GMM和K-means直观对比**

**GMM：**

* 先计算所有数据对每个分模型的响应度
* 根据响应度计算每个分模型的参数
* 迭代

**K-means：**

* 先计算所有数据对于K个点的距离，取距离最近的点作为自己所属于的类 
* 根据上一步的类别划分更新点的位置（点的位置就可以看做是模型参数）
* 迭代

可以看出GMM和K-means还是有很大的相同点的。GMM中数据对高斯分量的响应度就相当于K-means中的距离计算，GMM中的根据响应度计算高斯分量参数就相当于K-means中计算分类点的位置。然后它们都通过不断迭代达到最优。不同的是：GMM模型给出的是每一个观测点由哪个高斯分量生成的概率，而K-means直接给出一个观测点属于哪一类。




### 2. 开源语音识别项目 ###

1. [PaddlePaddle/DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech)
2. [ASRT A Deep-Learning-Based Chinese Speech Recognition System_SpeechRecognition](https://github.com/nl8590687/ASRT_SpeechRecognition)
3. [ESPnet: end-to-end speech processing toolkit](https://github.com/espnet/espnet)
4. [Automatic Speech Recognition in Tensorflow 2](https://github.com/TensorSpeech/TensorFlowASR)




### Reference ###

1. [隐马尔科夫模型（HMM）](https://blog.csdn.net/qq_27586341/article/details/94602772)
2. [高斯混合模型（GMM）介绍以及学习笔记](https://blog.csdn.net/jojozhangju/article/details/19182013)
3. [高斯混合模型的终极理解](https://blog.csdn.net/xmu_jupiter/article/details/50889023)
