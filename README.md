# Simple bbdm deep learning's code framework

## Requirements
* Python 3.5
* Keras 2.2
* Tensorflow 1.12
* Numpy 1.15
* scikit-learn 0.20

## Detial
* ``data_onehot.py`` 用来对原始sequences以及label进行onehot编码,并转换成数组形式方便加载。
* ``data_cv.py`` 加载上面的数据，划分训练集和测试集，并做交叉验证划分。
* ``data_model.py`` 用keras堆积你的模型结构，可以根据实际需求重新定义。
* ``data_metrics.py`` 计算评估指标：包括``aupr``,``auc``,``f1``,``accuracy``,``recall``,``specision``,``precision``。
* ``data_parameters.py`` 所有的参数包括超参数都在这里，如果你调参，只需要在这里更改，然后直接运行代码就好了。
* ``data_main.py`` 训练过程，并将最终的结果输出到csv文件里。
* ``data_util.py`` 包括一些可视化函数。
* ``data``        数据文件夹。
* ``image``       保存可视化的图片到该文件夹。

## Usage
先激活tensorflow环境，在该环境运行代码如下（这里以人类数据集为例）：  
``python data_main.py ./data/1vs1/Human_posi_samples.txt ./data/1vs1/Human_nega_samples.txt``  

## FAQ
* 1.怎样使用你的数据？  
由于各个数据形式不一样，你需要重新写一份``data_onehot.py``文件，工作包括填充``sequences``和``label``,onehot编码，其它细节可参考上述代码。  
* 2.怎样更改模型？  
在``data_model.py``更改模型结构，注意数据维度可能也要改变。你也可以用model.add()的方式更改模型，由于这里需要多输入以及添加特征等，写成了上述形式。  
* 3.怎样调参？  
在``data_parameters.py``修改参数即可。  
* 4.打印模型结构报错？  
打印模型结构可能会报错，例如image文件夹下model.png是当前的模型结构。装这个依赖包稍微复杂一点，如果你嫌麻烦，你可以在``data_model.py``注释掉下面这一行。  
``data_util.visual_model(model)``  
* 5.其它？    
Gpu使用，绘图啊等等，说到底你还的看一遍代码。。。  
## End
这个代码写的时间很短，仅上传了大部分代码，加特征的代码完成后再加上去。当然你可以继续完善：
* Gridsearch
* others
