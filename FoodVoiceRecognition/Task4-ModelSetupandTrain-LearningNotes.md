### 1. Notebook ###

运行结果： [Task4-ModelSetupandTrain.ipynb](https://github.com/frankyangdev/NLP-Learning/blob/main/FoodVoiceRecognition/Task4-ModelSetupandTrain.ipynb)

### 2. Convolution Neural Network (CNN) ###

CNN由纽约大学的Yann LeCun于1998年提出。CNN本质上是一个多层感知机，其成功的原因关键在于它所采用的局部连接和共享权值的方式，一方面减少了的权值的数量使得网络易于优化，另一方面降低了过拟合的风险。CNN是神经网络中的一种，它的权值共享网络结构使之更类似于生物神经网络，降低了网络模型的复杂度，减少了权值的数量。该优点在网络的输入是多维图像时表现的更为明显，使图像可以直接作为网络的输入，避免了传统识别算法中复杂的特征提取和数据重建过程。在二维图像处理上有众多优势，如网络能自行抽取图像特征包括颜色、纹理、形状及图像的拓扑结构；在处理二维图像问题上，特别是识别位移、缩放及其它形式扭曲不变性的应用上具有良好的鲁棒性和运算效率等。

CNN本身可以采用不同的神经元和学习规则的组合形式。

CNN具有一些传统技术所没有的优点：良好的容错能力、并行处理能力和自学习能力，可处理环境信息复杂，背景知识不清楚，推理规则不明确情况下的问题，允许样品有较大的缺损、畸变，运行速度快，自适应性能好，具有较高的分辨率。它是通过结构重组和减少权值将特征抽取功能融合进多层感知器，省略识别前复杂的图像特征抽取过程。
 
卷积神经网络属于前馈网络的一种，是一种专门处理类似网格数据的神经网络，其特点就是每一层神经元只响应前一层的局部范围内的神经元。
卷积网络一般由：卷积运算+非线性操作（RELU）+池化 +若干全连接层。



[]https://blog.csdn.net/fengbingchun/article/details/50529500)