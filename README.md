# **一个标签分类Pipeline**

```
核心思想参见./doc/pipeline.pptx
```


### 组成部分包括：

* 标注（此git项目未涉及）
* 数据预处理（./data/）
* 模型（./models/）
* 评价指标（./utils/）
* Demo（./demo/）


### 项目目的：

0.这个项目是为了规范代码之用，工程项目不宜出现实验室风格的代码。

1.统一输入，避免各人重复进行数据预处理。

2.统一输出，方便量化各个模型的好坏。

3.模型和参数分离，实现模型复用。

4.pipeline模式直接实现demo。

5.各个环节以API形式连接，通过callback函数增加功能。期望打造社区风格的代码管理。

### 具体部分：

./data/

* 实现数据预处理，并存储。
* i.从excel文件获取文本和标签。
* ii.对文本进行切分（分词，分词等）。
* iii.文本和ID映射。
* iv.文本char/word embedding生成。


./models/

* 各个模型（包括但不限于Tensorflow模型）
* 模型只需要输出logits。


./utils/

* 评估方法（Accuracy，Precission，Recall）
* 调节logits阈值改进用户体验。
* 模型导出


./demo/

* 输出的训练好的模型。


### 其它文件

./benchmark

* 任务评测结果

./config/

* 各个任务的模型参数

./doc/

* 文档

./logs/

* 临时的log

./run/

* 运行任务

# demo logs

* demo ver1.0: 较之前改进ui,业务处理逻辑
