## 四、SVM支持向量机

### 1、代价函数
- 在逻辑回归中，我们的代价为：   
![\cos t({h_\theta }(x),y) = \left\{ {\begin{array}{c}    { - \log ({h_\theta }(x))} \\    { - \log (1 - {h_\theta }(x))}  \end{array} \begin{array}{c}    {y = 1} \\    {y = 0}  \end{array} } \right.](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Ccos%20t%28%7Bh_%5Ctheta%20%7D%28x%29%2Cy%29%20%3D%20%5Cleft%5C%7B%20%7B%5Cbegin%7Barray%7D%7Bc%7D%20%20%20%20%7B%20-%20%5Clog%20%28%7Bh_%5Ctheta%20%7D%28x%29%29%7D%20%5C%5C%20%20%20%20%7B%20-%20%5Clog%20%281%20-%20%7Bh_%5Ctheta%20%7D%28x%29%29%7D%20%20%5Cend%7Barray%7D%20%5Cbegin%7Barray%7D%7Bc%7D%20%20%20%20%7By%20%3D%201%7D%20%5C%5C%20%20%20%20%7By%20%3D%200%7D%20%20%5Cend%7Barray%7D%20%7D%20%5Cright.)，    
其中：![{h_\theta }({\text{z}}) = \frac{1}{{1 + {e^{ - z}}}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bh_%5Ctheta%20%7D%28%7B%5Ctext%7Bz%7D%7D%29%20%3D%20%5Cfrac%7B1%7D%7B%7B1%20%2B%20%7Be%5E%7B%20-%20z%7D%7D%7D%7D)，![z = {\theta ^T}x](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=z%20%3D%20%7B%5Ctheta%20%5ET%7Dx)
- 如图所示，如果`y=1`，`cost`代价函数如图所示    
![enter description here][24]    
我们想让![{\theta ^T}x &gt;  &gt; 0](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Ctheta%20%5ET%7Dx%20%3E%20%20%3E%200)，即`z>>0`，这样的话`cost`代价函数才会趋于最小（这是我们想要的），所以用途中**红色**的函数![\cos {t_1}(z)](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Ccos%20%7Bt_1%7D%28z%29)代替逻辑回归中的cost
- 当`y=0`时同样，用![\cos {t_0}(z)](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Ccos%20%7Bt_0%7D%28z%29)代替
![enter description here][25]
- 最终得到的代价函数为：    
![J(\theta ) = C\sum\limits_{i = 1}^m {[{y^{(i)}}\cos {t_1}({\theta ^T}{x^{(i)}}) + (1 - {y^{(i)}})\cos {t_0}({\theta ^T}{x^{(i)}})} ] + \frac{1}{2}\sum\limits_{j = 1}^{\text{n}} {\theta _j^2} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5Ctheta%20%29%20%3D%20C%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%7By%5E%7B%28i%29%7D%7D%5Ccos%20%7Bt_1%7D%28%7B%5Ctheta%20%5ET%7D%7Bx%5E%7B%28i%29%7D%7D%29%20%2B%20%281%20-%20%7By%5E%7B%28i%29%7D%7D%29%5Ccos%20%7Bt_0%7D%28%7B%5Ctheta%20%5ET%7D%7Bx%5E%7B%28i%29%7D%7D%29%7D%20%5D%20%2B%20%5Cfrac%7B1%7D%7B2%7D%5Csum%5Climits_%7Bj%20%3D%201%7D%5E%7B%5Ctext%7Bn%7D%7D%20%7B%5Ctheta%20_j%5E2%7D%20)   
最后我们想要![\mathop {\min }\limits_\theta  J(\theta )](http://latex.codecogs.com/gif.latex?%5Clarge%20%5Cmathop%20%7B%5Cmin%20%7D%5Climits_%5Ctheta%20J%28%5Ctheta%20%29)
- 之前我们逻辑回归中的代价函数为：   
![J(\theta ) =  - \frac{1}{m}\sum\limits_{i = 1}^m {[{y^{(i)}}\log ({h_\theta }({x^{(i)}}) + (1 - } {y^{(i)}})\log (1 - {h_\theta }({x^{(i)}})] + \frac{\lambda }{{2m}}\sum\limits_{j = 1}^n {\theta _j^2} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5Ctheta%20%29%20%3D%20%20-%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%7By%5E%7B%28i%29%7D%7D%5Clog%20%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20%2B%20%281%20-%20%7D%20%7By%5E%7B%28i%29%7D%7D%29%5Clog%20%281%20-%20%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%5D%20%2B%20%5Cfrac%7B%5Clambda%20%7D%7B%7B2m%7D%7D%5Csum%5Climits_%7Bj%20%3D%201%7D%5En%20%7B%5Ctheta%20_j%5E2%7D%20)   
可以认为这里的![C = \frac{m}{\lambda }](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=C%20%3D%20%5Cfrac%7Bm%7D%7B%5Clambda%20%7D)，只是表达形式问题，这里`C`的值越大，SVM的决策边界的`margin`也越大，下面会说明

### 2、Large Margin
- 如下图所示,SVM分类会使用最大的`margin`将其分开    
![enter description here][26]
- 先说一下向量内积
 - ![u = \left[ {\begin{array}{c}    {{u_1}} \\    {{u_2}}  \end{array} } \right]](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=u%20%3D%20%5Cleft%5B%20%7B%5Cbegin%7Barray%7D%7Bc%7D%20%20%20%20%7B%7Bu_1%7D%7D%20%5C%5C%20%20%20%20%7B%7Bu_2%7D%7D%20%20%5Cend%7Barray%7D%20%7D%20%5Cright%5D)，![v = \left[ {\begin{array}{c}    {{v_1}} \\    {{v_2}}  \end{array} } \right]](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=v%20%3D%20%5Cleft%5B%20%7B%5Cbegin%7Barray%7D%7Bc%7D%20%20%20%20%7B%7Bv_1%7D%7D%20%5C%5C%20%20%20%20%7B%7Bv_2%7D%7D%20%20%5Cend%7Barray%7D%20%7D%20%5Cright%5D)    
 - ![||u||](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7C%7Cu%7C%7C)表示`u`的**欧几里得范数**（欧式范数），![||u||{\text{ = }}\sqrt {{\text{u}}_1^2 + u_2^2} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7C%7Cu%7C%7C%7B%5Ctext%7B%20%3D%20%7D%7D%5Csqrt%20%7B%7B%5Ctext%7Bu%7D%7D_1%5E2%20%2B%20u_2%5E2%7D%20)
 - `向量V`在`向量u`上的投影的长度记为`p`，则：向量内积：    
 ![{{\text{u}}^T}v = p||u|| = {u_1}{v_1} + {u_2}{v_2}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%7B%5Ctext%7Bu%7D%7D%5ET%7Dv%20%3D%20p%7C%7Cu%7C%7C%20%3D%20%7Bu_1%7D%7Bv_1%7D%20%2B%20%7Bu_2%7D%7Bv_2%7D)      
 ![enter description here][27]  
根据向量夹角公式推导一下即可，![\cos \theta  = \frac{{\overrightarrow {\text{u}} \overrightarrow v }}{{|\overrightarrow {\text{u}} ||\overrightarrow v |}}](http://latex.codecogs.com/gif.latex?%5Clarge%20%5Ccos%20%5Ctheta%20%3D%20%5Cfrac%7B%7B%5Coverrightarrow%20%7B%5Ctext%7Bu%7D%7D%20%5Coverrightarrow%20v%20%7D%7D%7B%7B%7C%5Coverrightarrow%20%7B%5Ctext%7Bu%7D%7D%20%7C%7C%5Coverrightarrow%20v%20%7C%7D%7D)

- 前面说过，当`C`越大时，`margin`也就越大，我们的目的是最小化代价函数`J(θ)`,当`margin`最大时，`C`的乘积项![\sum\limits_{i = 1}^m {[{y^{(i)}}\cos {t_1}({\theta ^T}{x^{(i)}}) + (1 - {y^{(i)}})\cos {t_0}({\theta ^T}{x^{(i)}})} ]](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%7By%5E%7B%28i%29%7D%7D%5Ccos%20%7Bt_1%7D%28%7B%5Ctheta%20%5ET%7D%7Bx%5E%7B%28i%29%7D%7D%29%20%2B%20%281%20-%20%7By%5E%7B%28i%29%7D%7D%29%5Ccos%20%7Bt_0%7D%28%7B%5Ctheta%20%5ET%7D%7Bx%5E%7B%28i%29%7D%7D%29%7D%20%5D)要很小，所以近似为：   
![J(\theta ) = C0 + \frac{1}{2}\sum\limits_{j = 1}^{\text{n}} {\theta _j^2}  = \frac{1}{2}\sum\limits_{j = 1}^{\text{n}} {\theta _j^2}  = \frac{1}{2}(\theta _1^2 + \theta _2^2) = \frac{1}{2}{\sqrt {\theta _1^2 + \theta _2^2} ^2}](http://latex.codecogs.com/gif.latex?%5Clarge%20J%28%5Ctheta%20%29%20%3D%20C0%20&plus;%20%5Cfrac%7B1%7D%7B2%7D%5Csum%5Climits_%7Bj%20%3D%201%7D%5E%7B%5Ctext%7Bn%7D%7D%20%7B%5Ctheta%20_j%5E2%7D%20%3D%20%5Cfrac%7B1%7D%7B2%7D%5Csum%5Climits_%7Bj%20%3D%201%7D%5E%7B%5Ctext%7Bn%7D%7D%20%7B%5Ctheta%20_j%5E2%7D%20%3D%20%5Cfrac%7B1%7D%7B2%7D%28%5Ctheta%20_1%5E2%20&plus;%20%5Ctheta%20_2%5E2%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%7B%5Csqrt%20%7B%5Ctheta%20_1%5E2%20&plus;%20%5Ctheta%20_2%5E2%7D%20%5E2%7D)，      
我们最后的目的就是求使代价最小的`θ`
- 由   
![\left\{ {\begin{array}{c}    {{\theta ^T}{x^{(i)}} \geqslant 1} \\    {{\theta ^T}{x^{(i)}} \leqslant  - 1}  \end{array} } \right.\begin{array}{c}    {({y^{(i)}} = 1)} \\    {({y^{(i)}} = 0)}  \end{array} ](http://latex.codecogs.com/gif.latex?%5Clarge%20%5Cleft%5C%7B%20%7B%5Cbegin%7Barray%7D%7Bc%7D%20%7B%7B%5Ctheta%20%5ET%7D%7Bx%5E%7B%28i%29%7D%7D%20%5Cgeqslant%201%7D%20%5C%5C%20%7B%7B%5Ctheta%20%5ET%7D%7Bx%5E%7B%28i%29%7D%7D%20%5Cleqslant%20-%201%7D%20%5Cend%7Barray%7D%20%7D%20%5Cright.%5Cbegin%7Barray%7D%7Bc%7D%20%7B%28%7By%5E%7B%28i%29%7D%7D%20%3D%201%29%7D%20%5C%5C%20%7B%28%7By%5E%7B%28i%29%7D%7D%20%3D%200%29%7D%20%5Cend%7Barray%7D)可以得到：    
![\left\{ {\begin{array}{c}    {{p^{(i)}}||\theta || \geqslant 1} \\    {{p^{(i)}}||\theta || \leqslant  - 1}  \end{array} } \right.\begin{array}{c}    {({y^{(i)}} = 1)} \\    {({y^{(i)}} = 0)}  \end{array} ](http://latex.codecogs.com/gif.latex?%5Clarge%20%5Cleft%5C%7B%20%7B%5Cbegin%7Barray%7D%7Bc%7D%20%7B%7Bp%5E%7B%28i%29%7D%7D%7C%7C%5Ctheta%20%7C%7C%20%5Cgeqslant%201%7D%20%5C%5C%20%7B%7Bp%5E%7B%28i%29%7D%7D%7C%7C%5Ctheta%20%7C%7C%20%5Cleqslant%20-%201%7D%20%5Cend%7Barray%7D%20%7D%20%5Cright.%5Cbegin%7Barray%7D%7Bc%7D%20%7B%28%7By%5E%7B%28i%29%7D%7D%20%3D%201%29%7D%20%5C%5C%20%7B%28%7By%5E%7B%28i%29%7D%7D%20%3D%200%29%7D%20%5Cend%7Barray%7D)，`p`即为`x`在`θ`上的投影
- 如下图所示，假设决策边界如图，找其中的一个点，到`θ`上的投影为`p`,则![p||\theta || \geqslant 1](http://latex.codecogs.com/gif.latex?%5Clarge%20p%7C%7C%5Ctheta%20%7C%7C%20%5Cgeqslant%201)或者![p||\theta || \leqslant  - 1](http://latex.codecogs.com/gif.latex?%5Clarge%20p%7C%7C%5Ctheta%20%7C%7C%20%5Cleqslant%20-%201)，若是`p`很小，则需要![||\theta ||](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7C%7C%5Ctheta%20%7C%7C)很大，这与我们要求的`θ`使![||\theta || = \frac{1}{2}\sqrt {\theta _1^2 + \theta _2^2} ](http://latex.codecogs.com/gif.latex?%5Clarge%20%7C%7C%5Ctheta%20%7C%7C%20%3D%20%5Cfrac%7B1%7D%7B2%7D%5Csqrt%20%7B%5Ctheta%20_1%5E2%20&plus;%20%5Ctheta%20_2%5E2%7D)最小相违背，**所以**最后求的是`large margin`   
![enter description here][28]

### 3、SVM Kernel（核函数）
- 对于线性可分的问题，使用**线性核函数**即可
- 对于线性不可分的问题，在逻辑回归中，我们是将`feature`映射为使用多项式的形式![1 + {x_1} + {x_2} + x_1^2 + {x_1}{x_2} + x_2^2](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=1%20%2B%20%7Bx_1%7D%20%2B%20%7Bx_2%7D%20%2B%20x_1%5E2%20%2B%20%7Bx_1%7D%7Bx_2%7D%20%2B%20x_2%5E2)，`SVM`中也有**多项式核函数**，但是更常用的是**高斯核函数**，也称为**RBF核**
- 高斯核函数为：![f(x) = {e^{ - \frac{{||x - u|{|^2}}}{{2{\sigma ^2}}}}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=f%28x%29%20%3D%20%7Be%5E%7B%20-%20%5Cfrac%7B%7B%7C%7Cx%20-%20u%7C%7B%7C%5E2%7D%7D%7D%7B%7B2%7B%5Csigma%20%5E2%7D%7D%7D%7D%7D)     
假设如图几个点，
![enter description here][29]
令：   
![{f_1} = similarity(x,{l^{(1)}}) = {e^{ - \frac{{||x - {l^{(1)}}|{|^2}}}{{2{\sigma ^2}}}}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bf_1%7D%20%3D%20similarity%28x%2C%7Bl%5E%7B%281%29%7D%7D%29%20%3D%20%7Be%5E%7B%20-%20%5Cfrac%7B%7B%7C%7Cx%20-%20%7Bl%5E%7B%281%29%7D%7D%7C%7B%7C%5E2%7D%7D%7D%7B%7B2%7B%5Csigma%20%5E2%7D%7D%7D%7D%7D)   
![{f_2} = similarity(x,{l^{(2)}}) = {e^{ - \frac{{||x - {l^{(2)}}|{|^2}}}{{2{\sigma ^2}}}}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bf_2%7D%20%3D%20similarity%28x%2C%7Bl%5E%7B%282%29%7D%7D%29%20%3D%20%7Be%5E%7B%20-%20%5Cfrac%7B%7B%7C%7Cx%20-%20%7Bl%5E%7B%282%29%7D%7D%7C%7B%7C%5E2%7D%7D%7D%7B%7B2%7B%5Csigma%20%5E2%7D%7D%7D%7D%7D)
.
.
.
- 可以看出，若是`x`与![{l^{(1)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bl%5E%7B%281%29%7D%7D)距离较近，==》![{f_1} \approx {e^0} = 1](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bf_1%7D%20%5Capprox%20%7Be%5E0%7D%20%3D%201)，（即相似度较大）   
若是`x`与![{l^{(1)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bl%5E%7B%281%29%7D%7D)距离较远，==》![{f_2} \approx {e^{ - \infty }} = 0](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bf_2%7D%20%5Capprox%20%7Be%5E%7B%20-%20%5Cinfty%20%7D%7D%20%3D%200)，（即相似度较低）
- 高斯核函数的`σ`越小，`f`下降的越快      
![enter description here][30]
![enter description here][31]

- 如何选择初始的![{l^{(1)}}{l^{(2)}}{l^{(3)}}...](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bl%5E%7B%281%29%7D%7D%7Bl%5E%7B%282%29%7D%7D%7Bl%5E%7B%283%29%7D%7D...)
 - 训练集：![(({x^{(1)}},{y^{(1)}}),({x^{(2)}},{y^{(2)}}),...({x^{(m)}},{y^{(m)}}))](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%28%28%7Bx%5E%7B%281%29%7D%7D%2C%7By%5E%7B%281%29%7D%7D%29%2C%28%7Bx%5E%7B%282%29%7D%7D%2C%7By%5E%7B%282%29%7D%7D%29%2C...%28%7Bx%5E%7B%28m%29%7D%7D%2C%7By%5E%7B%28m%29%7D%7D%29%29)
 - 选择：![{l^{(1)}} = {x^{(1)}},{l^{(2)}} = {x^{(2)}}...{l^{(m)}} = {x^{(m)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bl%5E%7B%281%29%7D%7D%20%3D%20%7Bx%5E%7B%281%29%7D%7D%2C%7Bl%5E%7B%282%29%7D%7D%20%3D%20%7Bx%5E%7B%282%29%7D%7D...%7Bl%5E%7B%28m%29%7D%7D%20%3D%20%7Bx%5E%7B%28m%29%7D%7D)
 - 对于给出的`x`，计算`f`,令：![f_0^{(i)} = 1](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=f_0%5E%7B%28i%29%7D%20%3D%201)所以：![{f^{(i)}} \in {R^{m + 1}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bf%5E%7B%28i%29%7D%7D%20%5Cin%20%7BR%5E%7Bm%20%2B%201%7D%7D)
 - 最小化`J`求出`θ`，          
 ![J(\theta ) = C\sum\limits_{i = 1}^m {[{y^{(i)}}\cos {t_1}({\theta ^T}{f^{(i)}}) + (1 - {y^{(i)}})\cos {t_0}({\theta ^T}{f^{(i)}})} ] + \frac{1}{2}\sum\limits_{j = 1}^{\text{n}} {\theta _j^2} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5Ctheta%20%29%20%3D%20C%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%7By%5E%7B%28i%29%7D%7D%5Ccos%20%7Bt_1%7D%28%7B%5Ctheta%20%5ET%7D%7Bf%5E%7B%28i%29%7D%7D%29%20%2B%20%281%20-%20%7By%5E%7B%28i%29%7D%7D%29%5Ccos%20%7Bt_0%7D%28%7B%5Ctheta%20%5ET%7D%7Bf%5E%7B%28i%29%7D%7D%29%7D%20%5D%20%2B%20%5Cfrac%7B1%7D%7B2%7D%5Csum%5Climits_%7Bj%20%3D%201%7D%5E%7B%5Ctext%7Bn%7D%7D%20%7B%5Ctheta%20_j%5E2%7D%20)
 - 如果![{\theta ^T}f \geqslant 0](http://latex.codecogs.com/gif.latex?%5Clarge%20%7B%5Ctheta%20%5ET%7Df%20%5Cgeqslant%200)，==》预测`y=1`

### 4、使用`scikit-learn`中的`SVM`模型代码
- [全部代码](../code/4-SVM/SVM_scikit-learn.py)
- 线性可分的,指定核函数为`linear`：
```
    '''data1——线性分类'''
    data1 = spio.loadmat('data1.mat')
    X = data1['X']
    y = data1['y']
    y = np.ravel(y)
    plot_data(X,y)
    
    model = svm.SVC(C=1.0,kernel='linear').fit(X,y) # 指定核函数为线性核函数
```
- 非线性可分的，默认核函数为`rbf`
```
    '''data2——非线性分类'''
    data2 = spio.loadmat('data2.mat')
    X = data2['X']
    y = data2['y']
    y = np.ravel(y)
    plt = plot_data(X,y)
    plt.show()
    
    model = svm.SVC(gamma=100).fit(X,y)     # gamma为核函数的系数，值越大拟合的越好
```
### 5、运行结果
- 线性可分的决策边界：    
![enter description here][32]
- 线性不可分的决策边界：   
![enter description here][33]

--------------------------

  [24]: ./../images/SVM_01.png "SVM_01.png"
  [25]: ./../images/SVM_02.png "SVM_02.png"
  [26]: ./../images/SVM_03.png "SVM_03.png"
  [27]: ./../images/SVM_04.png "SVM_04.png"
  [28]: ./../images/SVM_05.png "SVM_05.png"
  [29]: ./../images/SVM_06.png "SVM_06.png"
  [30]: ./../images/SVM_07.png "SVM_07.png"
  [31]: ./../images/SVM_08.png "SVM_08.png"
  [32]: ./../images/SVM_09.png "SVM_09.png"
  [33]: ./../images/SVM_10.png "SVM_10.png"
  