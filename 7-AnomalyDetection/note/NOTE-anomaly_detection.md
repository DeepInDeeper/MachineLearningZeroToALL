## 七、异常检测 Anomaly Detection
- [全部代码](../../code/7-AnomalyDetection/AnomalyDetection.py)

### 1、高斯分布（正态分布）`Gaussian distribution` 
- 分布函数：![$$p(x) = {1 \over {\sqrt {2\pi } \sigma }}{e^{ - {{{{(x - u)}^2}} \over {2{\sigma ^2}}}}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24p%28x%29%20%3D%20%7B1%20%5Cover%20%7B%5Csqrt%20%7B2%5Cpi%20%7D%20%5Csigma%20%7D%7D%7Be%5E%7B%20-%20%7B%7B%7B%7B%28x%20-%20u%29%7D%5E2%7D%7D%20%5Cover%20%7B2%7B%5Csigma%20%5E2%7D%7D%7D%7D%7D%24%24)
 - 其中，`u`为数据的**均值**，`σ`为数据的**标准差**
 - `σ`越**小**，对应的图像越**尖**
- 参数估计（`parameter estimation`）
 - ![$$u = {1 \over m}\sum\limits_{i = 1}^m {{x^{(i)}}} $$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24u%20%3D%20%7B1%20%5Cover%20m%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7Bx%5E%7B%28i%29%7D%7D%7D%20%24%24)
 - ![$${\sigma ^2} = {1 \over m}\sum\limits_{i = 1}^m {{{({x^{(i)}} - u)}^2}} $$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%7B%5Csigma%20%5E2%7D%20%3D%20%7B1%20%5Cover%20m%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7B%7B%28%7Bx%5E%7B%28i%29%7D%7D%20-%20u%29%7D%5E2%7D%7D%20%24%24)

### 2、异常检测算法
- 例子
 - 训练集：![$$\{ {x^{(1)}},{x^{(2)}}, \cdots {x^{(m)}}\} $$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%5C%7B%20%7Bx%5E%7B%281%29%7D%7D%2C%7Bx%5E%7B%282%29%7D%7D%2C%20%5Ccdots%20%7Bx%5E%7B%28m%29%7D%7D%5C%7D%20%24%24),其中![$$x \in {R^n}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24x%20%5Cin%20%7BR%5En%7D%24%24)
 - 假设![$${x_1},{x_2} \cdots {x_n}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%7Bx_1%7D%2C%7Bx_2%7D%20%5Ccdots%20%7Bx_n%7D%24%24)相互独立，建立model模型：![$$p(x) = p({x_1};{u_1},\sigma _1^2)p({x_2};{u_2},\sigma _2^2) \cdots p({x_n};{u_n},\sigma _n^2) = \prod\limits_{j = 1}^n {p({x_j};{u_j},\sigma _j^2)} $$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24p%28x%29%20%3D%20p%28%7Bx_1%7D%3B%7Bu_1%7D%2C%5Csigma%20_1%5E2%29p%28%7Bx_2%7D%3B%7Bu_2%7D%2C%5Csigma%20_2%5E2%29%20%5Ccdots%20p%28%7Bx_n%7D%3B%7Bu_n%7D%2C%5Csigma%20_n%5E2%29%20%3D%20%5Cprod%5Climits_%7Bj%20%3D%201%7D%5En%20%7Bp%28%7Bx_j%7D%3B%7Bu_j%7D%2C%5Csigma%20_j%5E2%29%7D%20%24%24)
- 过程
 - 选择具有代表异常的`feature`:xi
 - 参数估计：![$${u_1},{u_2}, \cdots ,{u_n};\sigma _1^2,\sigma _2^2 \cdots ,\sigma _n^2$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%7Bu_1%7D%2C%7Bu_2%7D%2C%20%5Ccdots%20%2C%7Bu_n%7D%3B%5Csigma%20_1%5E2%2C%5Csigma%20_2%5E2%20%5Ccdots%20%2C%5Csigma%20_n%5E2%24%24)
 - 计算`p(x)`,若是`P(x)<ε`则认为异常，其中`ε`为我们要求的概率的临界值`threshold`
- 这里只是**单元高斯分布**，假设了`feature`之间是独立的，下面会讲到**多元高斯分布**，会自动捕捉到`feature`之间的关系
- **参数估计**实现代码
```
# 参数估计函数（就是求均值和方差）
def estimateGaussian(X):
    m,n = X.shape
    mu = np.zeros((n,1))
    sigma2 = np.zeros((n,1))
    
    mu = np.mean(X, axis=0) # axis=0表示列，每列的均值
    sigma2 = np.var(X,axis=0) # 求每列的方差
    return mu,sigma2
```

### 3、评价`p(x)`的好坏，以及`ε`的选取
- 对**偏斜数据**的错误度量
 - 因为数据可能是非常**偏斜**的（就是`y=1`的个数非常少，(`y=1`表示异常)），所以可以使用`Precision/Recall`，计算`F1Score`(在**CV交叉验证集**上)
 - 例如：预测癌症，假设模型可以得到`99%`能够预测正确，`1%`的错误率，但是实际癌症的概率很小，只有`0.5%`，那么我们始终预测没有癌症y=0反而可以得到更小的错误率。使用`error rate`来评估就不科学了。
 - 如下图记录：    
 ![enter description here][49]
 - ![$$\Pr ecision = {{TP} \over {TP + FP}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%5CPr%20ecision%20%3D%20%7B%7BTP%7D%20%5Cover%20%7BTP%20&plus;%20FP%7D%7D%24%24) ，即：**正确预测正样本/所有预测正样本**
 - ![$${\mathop{\rm Re}\nolimits} {\rm{call}} = {{TP} \over {TP + FN}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%7B%5Cmathop%7B%5Crm%20Re%7D%5Cnolimits%7D%20%7B%5Crm%7Bcall%7D%7D%20%3D%20%7B%7BTP%7D%20%5Cover%20%7BTP%20&plus;%20FN%7D%7D%24%24) ，即：**正确预测正样本/真实值为正样本**
 - 总是让`y=1`(较少的类)，计算`Precision`和`Recall`
 - ![$${F_1}Score = 2{{PR} \over {P + R}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%7BF_1%7DScore%20%3D%202%7B%7BPR%7D%20%5Cover%20%7BP%20&plus;%20R%7D%7D%24%24)
 - 还是以癌症预测为例，假设预测都是no-cancer，TN=199，FN=1，TP=0，FP=0，所以：Precision=0/0，Recall=0/1=0，尽管accuracy=199/200=99.5%，但是不可信。

- `ε`的选取
 - 尝试多个`ε`值，使`F1Score`的值高
- 实现代码
```
# 选择最优的epsilon，即：使F1Score最大    
def selectThreshold(yval,pval):
    '''初始化所需变量'''
    bestEpsilon = 0.
    bestF1 = 0.
    F1 = 0.
    step = (np.max(pval)-np.min(pval))/1000
    '''计算'''
    for epsilon in np.arange(np.min(pval),np.max(pval),step):
        cvPrecision = pval<epsilon
        tp = np.sum((cvPrecision == 1) & (yval == 1)).astype(float)  # sum求和是int型的，需要转为float
        fp = np.sum((cvPrecision == 1) & (yval == 0)).astype(float)
        fn = np.sum((cvPrecision == 1) & (yval == 0)).astype(float)
        precision = tp/(tp+fp)  # 精准度
        recision = tp/(tp+fn)   # 召回率
        F1 = (2*precision*recision)/(precision+recision)  # F1Score计算公式
        if F1 > bestF1:  # 修改最优的F1 Score
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon,bestF1
```

### 4、选择使用什么样的feature（单元高斯分布）
- 如果一些数据不是满足高斯分布的，可以变化一下数据，例如`log(x+C),x^(1/2)`等
- 如果`p(x)`的值无论异常与否都很大，可以尝试组合多个`feature`,(因为feature之间可能是有关系的)

### 5、多元高斯分布
- 单元高斯分布存在的问题
 - 如下图，红色的点为异常点，其他的都是正常点（比如CPU和memory的变化）   
 ![enter description here][50]
 - x1对应的高斯分布如下：   
 ![enter description here][51]
 - x2对应的高斯分布如下：   
 ![enter description here][52]
 - 可以看出对应的p(x1)和p(x2)的值变化并不大，就不会认为异常
 - 因为我们认为feature之间是相互独立的，所以如上图是以**正圆**的方式扩展
- 多元高斯分布
 - ![$$x \in {R^n}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24x%20%5Cin%20%7BR%5En%7D%24%24)，并不是建立`p(x1),p(x2)...p(xn)`，而是统一建立`p(x)`
 - 其中参数：![$$\mu  \in {R^n},\Sigma  \in {R^{n \times {\rm{n}}}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%5Cmu%20%5Cin%20%7BR%5En%7D%2C%5CSigma%20%5Cin%20%7BR%5E%7Bn%20%5Ctimes%20%7B%5Crm%7Bn%7D%7D%7D%7D%24%24),`Σ`为**协方差矩阵**
 - ![$$p(x) = {1 \over {{{(2\pi )}^{{n \over 2}}}|\Sigma {|^{{1 \over 2}}}}}{e^{ - {1 \over 2}{{(x - u)}^T}{\Sigma ^{ - 1}}(x - u)}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24p%28x%29%20%3D%20%7B1%20%5Cover%20%7B%7B%7B%282%5Cpi%20%29%7D%5E%7B%7Bn%20%5Cover%202%7D%7D%7D%7C%5CSigma%20%7B%7C%5E%7B%7B1%20%5Cover%202%7D%7D%7D%7D%7D%7Be%5E%7B%20-%20%7B1%20%5Cover%202%7D%7B%7B%28x%20-%20u%29%7D%5ET%7D%7B%5CSigma%20%5E%7B%20-%201%7D%7D%28x%20-%20u%29%7D%7D%24%24)
 - 同样，`|Σ|`越小，`p(x)`越尖
 - 例如：    
 ![enter description here][53]，  
 表示x1,x2**正相关**，即x1越大，x2也就越大，如下图，也就可以将红色的异常点检查出了
 ![enter description here][54]      
 若：   
  ![enter description here][55]，   
 表示x1,x2**负相关**
- 实现代码：
```
# 多元高斯分布函数    
def multivariateGaussian(X,mu,Sigma2):
    k = len(mu)
    if (Sigma2.shape[0]>1):
        Sigma2 = np.diag(Sigma2)
    '''多元高斯分布函数'''    
    X = X-mu
    argu = (2*np.pi)**(-k/2)*np.linalg.det(Sigma2)**(-0.5)
    p = argu*np.exp(-0.5*np.sum(np.dot(X,np.linalg.inv(Sigma2))*X,axis=1))  # axis表示每行
    return p
```
### 6、单元和多元高斯分布特点
- 单元高斯分布
 - 人为可以捕捉到`feature`之间的关系时可以使用
 - 计算量小
- 多元高斯分布
 - 自动捕捉到相关的feature
 - 计算量大，因为：![$$\Sigma  \in {R^{n \times {\rm{n}}}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%5CSigma%20%5Cin%20%7BR%5E%7Bn%20%5Ctimes%20%7B%5Crm%7Bn%7D%7D%7D%7D%24%24)
 - `m>n`或`Σ`可逆时可以使用。（若不可逆，可能有冗余的x，因为线性相关，不可逆，或者就是m<n）

### 7、程序运行结果
- 显示数据     
![enter description here][56]
- 等高线      
![enter description here][57]
- 异常点标注   
![enter description here][58]



----------------------------------

  [49]: ../../images/AnomalyDetection_01.png "AnomalyDetection_01.png"
  [50]: ../../images/AnomalyDetection_04.png "AnomalyDetection_04.png"
  [51]: ../../images/AnomalyDetection_02.png "AnomalyDetection_02.png"
  [52]: ../../images/AnomalyDetection_03.png "AnomalyDetection_03.png"
  [53]: ../../images/AnomalyDetection_05.png "AnomalyDetection_05.png"
  [54]: ../../images/AnomalyDetection_07.png "AnomalyDetection_07.png"
  [55]: ../../images/AnomalyDetection_06.png "AnomalyDetection_06.png"
  [56]: ../../images/AnomalyDetection_08.png "AnomalyDetection_08.png"
  [57]: ../../images/AnomalyDetection_09.png "AnomalyDetection_09.png"
  [58]: ../../images/AnomalyDetection_10.png "AnomalyDetection_10.png"
