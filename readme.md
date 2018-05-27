放在前面，更新说明：
(TODO) ipynb脚本需要添加必要的解释和说明，最好能够完成笔记和代码相结合。
> 更新代码，能在python2 和python3 运行  
> 章节更改为ipynb，用来更好显示运行结果，方便查看。并在后期用来更好转换为pdf、html等格式，同时也添加更为详细的解释
> 原所有py文件修改，完善放置[code](code)文件夹下面，数据单独放在[data](data)文件夹下  
> readme.md 分章节下面，笔记增添 [LSJU机器学习笔记](https://github.com/zlotus/notes-LSJU-machine-learning)  
> （TODO） 增添吴恩达机器学习课程python代码  

`Github` 加载 `.ipynb` 的速度较慢，建议在[这里](http://nbviewer.jupyter.org/github/Peterchenyijie/MachineLearningZeroToALL/blob/master/readme.ipynb)中查看该项目。

requirement：
* scikit-learn >=0.19  
 
[![MIT license](https://img.shields.io/dub/l/vibe-d.svg)](https://github.com/PeterChenYijie/MachineLearningZeroToALL/blob/master/LICENSE)

机器学习算法Python实现
=========


## ipynb 学习目录
* [0 Basic Concept|基础概念](0-BasicConcept)
    * [概率论](0-BasicConcept/note/probability_theory.ipynb)
* [1 Linear Regression|线性回归](1-LinearRegression)
    * 1.1 [线性回归笔记](1-LinearRegression/note)
        * 1.1.1 [LSJU笔记-监督学习应用](1-LinearRegression/note/LSJU_chapter02.ipynb)
        * 1.1.2 [LSJU笔记-线性模型概率解释、局部加权回归](1-LinearRegression/note/LSJU-chapter03_01.ipynb)
        * 1.1.3 [LSJU笔记-一般线性模型](1-LinearRegression/note/LSJU_chapter04.ipynb)
        * 1.1.3 [笔记](1-LinearRegression/note/NOTE-linear_regression.ipynb)
    * 1.2 [线性回归的实现](1-LinearRegression/LinearRegression.ipynb)
    * 1.3 [使用sklearn 线性回归](1-LinearRegression/LinearRegression_sklearn.ipynb)
    * 1.4 [Mxnet 实现线性回归](1-LinearRegression/linear_regression_mxnet.ipynb)
* [2 Logistic Regression|逻辑回归](2-LogisticRegression)
    * 2.1 [逻辑回归笔记](2-LogisticRegression/note/)
        * 2.1.1 [LSJU笔记-分类问题、逻辑回归](2-LogisticRegression/note/LSJU-chapter03_2.ipynb)
        * 2.1.2 [逻辑回归笔记](2-LogisticRegression/note/NOTE-logistic_regression.ipynb)
    * 2.2 [逻辑回归的实现](2-LogisticRegression/LogisticRegression.ipynb)
    * 2.3 [使用sklearn 逻辑回归](2-LogisticRegression/LogisticRegression_scikit-learn.ipynb)
    * 2.4 [逻辑回归识别手写数字](2-LogisticRegression/LogisticRegression_OneVsAll.ipynb)
    * 2.5 [使用sklearn 逻辑回归识别手写数字](2-LogisticRegression/LogisticRegression_OneVsAll_scikit-learn.ipynb)
* [3 Neural Network|神经网络](3-NeuralNetwok)
    * 3.1 [神经网络笔记](3-NeuralNetwok/note/NOTE-neural_network.md)
    * 3.2 [神经网络识别手写数字](3-NeuralNetwok/NeuralNetwork.ipynb)
* [4 SVM|支持向量机](4-SVM)
    * 4.1 [SVM笔记](4-SVM/NOTE-SVM.md)
    * 4.2 [SVM实现]
    * 4.3 [使用sklearn SVM](4-SVM/SVM_scikit-learn.ipynb)
* [5 K-Means|聚类](5-K-Means)
    * 5.1 [K-Mean 聚类笔记](5-K-Means/LSJU----NOTE-K-Means.ipynb)
    * 5.2 [K-Mean的实现](5-K-Means/K-Means.ipynb)
    * 5.3 [使用sklearn K-Means](5-K-Means/K-Means-sklearn.ipynb)
* [6 PCA|主成分分析](6-PCA)
    * 6.1 [PCA笔记](6-PCA/note/LSJU----NOTE-PCA.ipynb)
    * 6.2 [PCA的实现](6-PCA/PCA.ipynb)
    * 6.3 [使用sklearn PCA](6-PCA/PCA_sklearn.ipynb)
* [7 Anomaly Detection|异常检测](7-AnomalyDetection)
    * 7.1 [异常检测笔记](7-AnomalyDetection/note/NOTE-anomaly_detection.md)
    * 7.2 [异常检测实现](7-AnomalyDetection/AnomalyDetection.ipynb)
* [8 HMM|隐马尔可夫模型](8-HMM)
    * 8.1 [马尔科夫决策过程](8-HMM/note/LSJU-HMM.ipynb)其他参考 [这里](8-HMM/note/sn06.ipynb)
* [9 NaiveBayer|朴素贝叶斯](9-NaiveBayer)
    * 9.1 [笔记](9-NaiveBayer/note)
        * 9.1.1 [LSJU-生成学习算法、高斯判别分布、朴素贝叶斯算法](9-NaiveBayer/note/LSJU-chapter05.ipynb)


[收藏！机器学习算法优缺点综述][Title-6] + 七月在线实验室  
![img-06-01][img-06-01]
* 正则化算法（Regularization Algorithms）
    * 例子
        * 岭回归（Ridge Regression）
        * 最小绝对收缩与选择算子（LASSO）
        * GLASSO
        * 弹性网络（Elastic Net）
        * 最小角回归（Least-Angle Regression）
    * 优点
        * 其惩罚会减少过拟合
        * 总会有解决方法
    * 缺点
        * 惩罚会造成欠拟合
        * 很难校准

* 集成算法（Ensemble Algorithms）
    * 例子
        * Boosting
        * Bootstrapped Aggregation（Bagging）
        * AdaBoost
        * 层叠泛化（Stacked Generalization）（blending）
        * 梯度推进机（Gradient Boosting Machines，GBM）
        * 梯度提升回归树（Gradient Boosted Regression Trees，GBRT）
        * 随机森林（Random Forest）
    * 优点
        * 当先最先进的预测几乎都使用了算法集成。它比使用单个模型预测出来的结果要精确的多
    * 缺点
        * 需要大量的维护工作
* 决策树算法（Decision Tree Algorithm）
    * 例子
        * 分类和回归树（Classification and Regression Tree，CART）
        * Iterative Dichotomiser 3（ID3）
        * C4.5 和 C5.0（一种强大方法的两个不同版本）
    * 优点
        * 容易解释
        * 非参数化
    * 缺点
        * 趋向过拟合
        * 可能或陷于局部最小值中
        * 没有在线学习

* 回归（Regression）
    * 例子
        * 普通最小二乘回归（Ordinary Least Squares Regression，OLSR）
        * 线性回归（Linear Regression）
        * 逻辑回归（Logistic Regression）
        * 逐步回归（Stepwise Regression）
        * 多元自适应回归样条（Multivariate Adaptive Regression Splines，MARS）
        * 本地散点平滑估计（Locally Estimated Scatterplot Smoothing，LOESS）
    * 优点
        * 直接、快速
        * 知名度高
    * 缺点
        * 要求严格的假设
        * 需要处理异常值

* 人工神经网络（Artificial Neural Network）
    * 例子
        * 感知器
        * 反向传播
        * Hopfield 网络
        * 径向基函数网络（Radial Basis Function Network，RBFN）
    * 优点：
        * 在语音、语义、视觉、各类游戏（如围棋）的任务中表现极好
        * 算法可以快速调整，适应新的问题
    * 缺点：
        * 需要大量数据进行训练
        * 训练要求很高的硬件配置
        * 模型处于「黑箱状态」，难以理解内部机制
        * 元参数（Metaparameter）与网络拓扑选择困难。

* 深度学习（Deep Learning）
    * 例子：
        * 深玻耳兹曼机（Deep Boltzmann Machine，DBM）
        * Deep Belief Networks（DBN）
        * 卷积神经网络（CNN）
        * Stacked Auto-Encoders
    * 优点/缺点：见神经网络

* 支持向量机（Support Vector Machine）
    * 优点
        * 在非线性可分问题上表现优秀
    * 缺点
        * 非常难以训练
        * 很难解释

* 降维算法（Dimensionality Reduction Algorithms）
    * 例子：
        * 主成分分析（Principal Component Analysis (PCA)）
        * 主成分回归（Principal Component Regression (PCR)）
        * 偏最小二乘回归（Partial Least Squares Regression (PLSR)）
        * Sammon 映射（Sammon Mapping）
        * 多维尺度变换（Multidimensional Scaling (MDS)）
        * 投影寻踪（Projection Pursuit）
        * 线性判别分析（Linear Discriminant Analysis (LDA)）
        * 混合判别分析（Mixture Discriminant Analysis (MDA)）
        * 二次判别分析（Quadratic Discriminant Analysis (QDA)）
        * 灵活判别分析（Flexible Discriminant Analysis (FDA)）
    * 优点：
        * 可处理大规模数据集
        * 无需在数据上进行假设
    * 缺点：
        * 难以搞定非线性数据
        * 难以理解结果的意义


* 聚类算法（Clustering Algorithms）
    * 例子：
        * K-均值（k-Means）
        * K-Medians 算法
        * Expectation–maximization（EM）
        * 分层集群（Hierarchical Clustering）
    * 优点：
        * 让数据变得有意义
    * 缺点：
        * 结果难以解读，针对不寻常的数据组，结果可能无用

* 基于实例的算法（Instance-based Algorithms）
    * 例子：
        * K 最近邻（k-Nearest Neighbor (kNN)）
        * 学习向量量化（Learning Vector Quantization (LVQ)）
        * 自组织映射（Self-Organizing Map (SOM)）
        * 局部加权学习（Locally Weighted Learning (LWL)）
    * 优点：
        * 算法简单、结果易于解读
    * 缺点：
        * 内存使用非常高
        * 计算成本高
        * 不可能用于高维特征空间

* 贝叶斯算法（Bayesian Algorithms）
    * 例子：
        * 朴素贝叶斯（Naive Bayer）
        * 高斯朴素贝叶斯（Gaussian Naive Bayer）
        * 多项式朴素贝叶斯（Multinomial Naive Bayer）
        * 平均一致依赖估计器（Averaged One—Dependence Estimators（AODE））
        * 贝叶斯网络（Bayeian Network（BN））
    * 优点
        * 快速、易于训练、给出了它们所需的资源能带来良好的表现
    * 缺点
        * 如果输入变量是相关的，则会出现问题

* 关联规则学习算法（Association Rule Learning Algorithms）
    * 例子
        * Apriori 算法（Apriori algorithm）
        * Eclat 算法（Eclat algorithm）
        * FP-growth

* 图模型（Graphical Models）
    * 例子：
        * 贝叶斯网络（Bayesian network）
        * 马尔可夫随机域（Markov random field）
        * 链图（Chain Graphs）
        * 祖先图（Ancestral graph）
    * 优点：
        * 模型清晰，能被直观地理解
    * 缺点：
        * 确定其依赖的拓扑很困难，有时候也很模糊

---
[Title-6]:https://mp.weixin.qq.com/s/Xomx6Z_fP1EsnwNEegj5Cw
[img-06-01]:img/20180527-06-01.jpg