# ljdlib
My data science lib
--
### 分箱方法
数据分箱的重要性离散特征的增加和减少都很容易，易于模型的快速迭代；稀疏向量内积乘法运算速度快，计算结果方便存储，容易扩展；离散化后的特征对异常数据有很强的鲁棒性：比如一个特征是年龄>30是1，否则0。如果特征没有离散化，一个异常数据“年龄300岁”会给模型造成很大的干扰；逻辑回归属于广义线性模型，表达能力受限；单变量离散化为N个后，每个变量有单独的权重，相当于为模型引入了非线性，能够提升模型表达能力，加大拟合；离散化后可以进行特征交叉，由M+N个变量变为M*N个变量，进一步引入非线性，提升表达能力；特征离散化后，模型会更稳定，比如如果对用户年龄离散化，20-30作为一个区间，不会因为一个用户年龄长了一岁就变成一个完全不同的人。当然处于区间相邻处的样本会刚好相反，所以怎么划分区间是门学问；特征离散化以后，起到了简化了逻辑回归模型的作用，降低了模型过拟合的风险。可以将缺失作为独立的一类带入模型。将所有变量变换到相似的尺度上。

作者：猪猪侠123
链接：https://zhuanlan.zhihu.com/p/240097602
来源：知乎
