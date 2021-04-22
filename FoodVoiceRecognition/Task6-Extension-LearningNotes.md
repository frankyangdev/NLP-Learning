### 1. HMM ###

**HMM的主要应用方向**

单分子动力学分析、密码分析、语音识别、语音合成、词性标记、扫描解决方案中的文档分离、机器翻译、局部放电、基因预测生物序列比对、时间序列分析、活动识别蛋白质折叠、变态病毒检测、DNA主题发现

**NLP方向：**

* 词性标注：给定一个词的序列（也就是句子），找出最可能的词性序列（标签是词性）。如ansj分词和ICTCLAS分词等。
* 分词：给定一个字的序列，找出最可能的标签序列（断句符号：[词尾]或[非词尾]构成的序列）。结巴分词目前就是利用BMES标签来分词的，B（开头）,M（中间),E(结尾),S(独立成词）
* 命名实体识别：给定一个词的序列，找出最可能的标签序列（内外符号：[内]表示词属于命名实体，[外]表示不属于）。如ICTCLAS实现的人名识别、翻译人名识别、地名识别都是用同一个Tagger实现的。
机器翻译




### 2. 开源语音识别项目 ###

1. [PaddlePaddle/DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech)
2. [ASRT A Deep-Learning-Based Chinese Speech Recognition System_SpeechRecognition](https://github.com/nl8590687/ASRT_SpeechRecognition)
3. [ESPnet: end-to-end speech processing toolkit](https://github.com/espnet/espnet)
4. [Automatic Speech Recognition in Tensorflow 2](https://github.com/TensorSpeech/TensorFlowASR)



### Reference ###

1. [隐马尔科夫模型（HMM）](https://blog.csdn.net/qq_27586341/article/details/94602772)
