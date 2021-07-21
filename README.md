# 基于softmax regression实现的文本多分类

项目已经在aistudio开源  
[基于softmax regression实现文本多分类](https://aistudio.baidu.com/aistudio/projectdetail/2135044)

softmax regression主要用于解决多元分类问题，本项目通过编写softmaxRegression类实现文本的多分类


```python
!unzip -oq /home/aistudio/data/data97334/movie-reviews.zip
```


```python
!unzip -oq /home/aistudio/train.tsv.zip
!unzip -oq /home/aistudio/test.tsv.zip
```


```python
!rm -rf train.tsv.zip
!rm -rf test.tsv.zip
```

## 四个py的介绍

###  1 feature_extraction.py
主要是两个模块的编写

特征表示：Bag-of-Word，N-gram

1. Bag-of-Word：词袋模型，根据语料建立词典vocab，词典中每个单词有一个index，M为词典的大小，将句子表示为一个M维向量，每一维的值对应该索引对应的单词在句子中出现与否或者出现的次数。这种特征表示不考虑单词出现的先后顺序，丢失了重要的语义信息。
2. N-gram：相比于词袋模型，N-gram将N个单词联合起来看作一个特征，例如2-gram，则语料库中所有两个挨着出现过的单词联合看作一个特征，相比于词袋模型，可把N-gram理解为在构建词典时将多个单词联合出现看作特征，最后构建特征向量时和词袋模型相同。这种特征表示考虑了部分单词的先后顺序，随着N的增大词典的规模会暴增，所以这样的处理方式不能捕获长程依赖。

###  2 data_preprocess.py

通过pandas读取文件

###  3 softmax_regerssion.py

编写SoftmaxRegression类

模型详解

输入数据X，维度NxM，每一行是一个样本，y是X中N个样本对应的标签，为了方便计算将标签转化为独热码，维度NxK，参数W，维度KxM，（下面用$W_i$ 表示$W[i]$），X中句子共有K个类别，第i 个样本被预测为c类的概率计算公式如下：

![](https://ai-studio-static-online.cdn.bcebos.com/605346e878c54f6cbbb98c2e83f36d3a1f0a2bcd8f284700bdbeb160a4e8c91f)

得到预测概率后，采用交叉熵损失函数计算损失

![](https://ai-studio-static-online.cdn.bcebos.com/6678be328e094752933730d60bc2f4bf3fdd221c3ea74c0ea8f56f510d5210cf)

loss对参数W求导得到：

![](https://ai-studio-static-online.cdn.bcebos.com/592a306a0547448da2f6d0fc95dd0c350ae8fb6a10ac4fae9905a7649b701949)

###  4 main.py

运行的主函数

## 开始训练


```python
!python main.py
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized
    Bow shape (1000, 363)
    Gram shape (1000, 968)
    epoch 0 loss [3.13428539]
    epoch 10 loss [1.7146469]
    epoch 20 loss [1.64050665]
    epoch 30 loss [1.5919535]
    epoch 40 loss [1.5830066]
    epoch 50 loss [1.39589994]
    epoch 60 loss [1.524077]
    epoch 70 loss [1.56308224]
    epoch 80 loss [1.49590111]
    epoch 90 loss [1.42763464]
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      if isinstance(obj, collections.Iterator):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2366: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return list(data) if isinstance(data, collections.MappingView) else data
    Figure(640x480)
    Bow train 0.83125 test 0.46
    epoch 0 loss [3.99405842]
    epoch 10 loss [1.11545373]
    epoch 20 loss [0.50951999]
    epoch 30 loss [0.49667043]
    epoch 40 loss [0.13800268]
    epoch 50 loss [0.09492191]
    epoch 60 loss [0.12789368]
    epoch 70 loss [0.02561636]
    epoch 80 loss [0.08942448]
    epoch 90 loss [0.05397729]
    Figure(640x480)
    Gram train 0.9975 test 0.515


![Bow](https://github.com/ITerydh/softmax-regression/blob/main/Bow.jpg)

![Gram](https://github.com/ITerydh/softmax-regression/blob/main/Gram.jpg)

# 总结

处理多分类任务时，通常使用Softmax Regression模型。

在神经网络中，如果问题是分类模型(即使是CNN或者RNN)，一般最后一层是Softmax Regression。

它的工作原理是将可以判定为某类的特征相加，然后将这些特征转化为判定是这一类的概率。

那么什么时候该用Softmax回归，又什么时候该用Logistc回归呢？当类别是严格互斥时用Softmax回归，当不是严格互斥，即某些数据可能属于多个类别时就用Logistic回归。

>全网同名: iterhui

我在AI Studio上获得钻石等级，点亮9个徽章，来互关呀~

>https://aistudio.baidu.com/aistudio/personalcenter/thirdview/643467
