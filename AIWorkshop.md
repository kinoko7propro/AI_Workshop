# AI Workshop
## Workshop1: Primer Concepts
### 多元智能理论
1. Linguistic Intelligence（语言智能）
2. Musical Intelligence（音乐智能）
3. Logical-Mathematical Intelligence（逻辑-数学智能）
4. Spatial Intelligence（空间智能）：感知、改变和重现视觉或空间信息的能力，包括构建3D图像和操纵它们的能力。
5. Bodily-Kinesthetic Intelligence（身体-动觉智能）
6. Intra-personal Intelligence（内省智能）
7. Interpersonal Intelligence（人际智能）
一个系统（例如人工智能）如果具备上述一种或多种智能，就可以被认为是人工智能。

### 人类智力的构成
- Reasoning 推理
- Learning 学习
- Problem Solving 问题解决
- Perception 感知
- Linguistic Intelligence 语言智力

#### Reasoning 
是一组过程，让我们排序，搜索，决策和评估
- Inductive Reasoning
- Deductive Reasoning
- Aductive Reasoning 

#### Learning 
- Aduditory Learning: 利用听觉
- Episdoic  Learning: 经历的事件
- Motor Learning: 肌肉的记忆
- Observational Learning: 模仿他人
- Perceptual Learning: 不断认知和记忆
- Relational Learning: 相对属性来刺激
1. Spatial Learning: 大脑内构造空间
2. Stimulus-Response Learning: 对刺激作出反应

#### Problem Solving 
采取路径进行感知或决策

#### Perception
将获取的信息整合在一起

#### Linguistic Intelligence 
使用，说，写语言

### AI 包含了什么？
#### Machine Learning
==Learning from the data==

#### Logic 
运用数理逻辑执行计算机程序

#### Searching
用于战略性的游戏，搜索空间

#### Artificial neural networks
高效的计算网络

#### Genetic Algorithm
其结果将基于对适应度最高的个体的选择

#### Knowledge Representation
用机器来呈现知识

### Simulating Human Thinking Procedure
几何 - 运动 - 身体 - 行为 - 认知

### Agent & Environment
- Agent: 通过传感器感知环境，通过效应器对环境采取行动


## Workshop2A: An overview of Machine Learning
==the ability to learn with the data and improve from experience==

### Supervised Machine Learning
- 结果已经知道，训练标注了正确答案
- 目标是将映射函数逼近的更好，当有新的输入数据时能做到预测
- Classification(分类): 进行分类输出
- Regression(回归): 输出真实数值
- *Decision tree, random forest, knn, regression*这些都是监督学习的算法

### Unsupevised Learning 
- 无正确答案
- Clustering(聚类): 用于发现数据中固有的分组
- Association(关联): 发现和描述数据的规则
- *K-means for clustering, Apriori algorithm for association*

### Reinforcement Machine Learning
不断使用试错法，从过去的经验中学习，获取更多知识来做出决策
*MDP, Actor-Critic*

### 常见的机器学习算法
#### Linear Regression线性回归
==输入变量和*单个*输出变量间存在线性关系==

#### Logistic Regression 逻辑回归
一种==分类==算法，根据自变量来估计0和1之类的离散值

#### Decision Tree 决策树
一种监督学习算法，用于==分类==

#### SVM
用于==分类==
使用二维空间中的支持向量将数据分为不同组

#### Naive Bayes朴素贝叶斯
一种==分类==技术
利用贝叶斯定理构建分类器，假设预测变量间是相互独立的

#### KNN K近邻算法
用于解决==分类==问题
储存可用案例，根据K个近邻的多数投票来对新案例进行分类，通过距离函数来进行衡量
距离函数: Euclidean, Minkowski and Hamming distance
标量需要标准化，否则会产生偏差
预处理需要去除噪声

#### K-Means Clustering
一种无监督学习，用于聚类
主要逻辑是通过对若干个簇对数据进行分类，关键词包括==centroid(质心)==, clsuter(簇)

#### Random Forest
一种监督学习，可用于分类、回归
它是一组决策树的集合，每种树会给出分类，而森林给出最佳分类
能处理缺失值，不会过拟合

## AI Workshop2B: Data Preparation 数据的预处理
- Numpy: 处理多维数组的同时不会牺牲速度
- Sklearn: 将原始特征向量转换为适合机器学习的表示形式

### Binarzation 二值化
将数值转换为布尔值，可以以某个数作为阈值，在如神经网络问题中我们需要二进制数据作为输入变量

### Mean Removal 均值去除
从特征向量中消除均值，使特征以0为中心，还能==去除偏差==

### Scaling 缩放
特征值可能在许多随机值间变化

### Normalization 正则化
对高维中心进行聚类
- L1 正则(归一)化: ==使得每行的绝对值之和不超过1==
- L2 正则化:  ==使得每行的平方和不超过1==

### 数据标注
标签编码: 文字标签转化为数字形式

## Workshop3: 监督学习--分类与回归
*基于==有标记的==训练数据构建机器学习模型*
### Classification 分类
*将数据排列到固定数量的类别，用于确定新数据点所属的类别*

### Naive Bayes Classifier(朴素贝叶斯分类器)
**Guassian, Multinomial, Bernoulli**

### SVM
将每个数据项在n维空间中绘制为一个点，特征的值代表其特定坐标，n为特征数量
***Kernal 内核***
- 一种支持向量机的技术，是接受低维空间转换为高维空间的函数

### logistic Regression
逻辑回归：解释输入变量和输出变量关系的数据，是一种回归，但常用于分类

### Confusion matrixes 混淆矩阵
一种用于描述分类器性能的图形或表格
对于二元分类而言：
|True Positive 真阳 |False Negative 假阴|
|:-----------------:|:-----------------|
|False Positive 假阳|True Negative 真阴 |


使用该矩阵可以直观了解模型是否被错误或混淆分类

### Regression
回归存在==无限多的==可能性
输入变量：自变量(independent)、预测变量(predictors)
输出变量：因变量(dependent)、标准变量(criterion variables)
当线性回归无法处理：ploynomial regression多项式回归
#### mean squared error(MSE) 均方误差: 误差的平方的平均数、预测值与实际值的平均平方差值
*总是非负的严格正数，越靠近0越好*
#### Explained variation 解释方差，表示对数据集中方差的解释程度，未被解释的则是未解释方差或残差方差
#### The coefficient of determination/R2 score 决定系数: 分析变量的差异并用第二个变量的差异来解释

## Workshop4: Predictive Analytics with Ensemble Learning: 集成学习下的预测分析
### Decision Tree 决策树
- 将数据集化为不同分支，经遍历后做出决策，每个节点都是一条决策规则
- entropy: 熵，对不确定性的度量
- 决策树的目标: 从根节点向叶节点移动的过程中减少不确定性
- 分类器的性能: 
1. **precision** 精度:  表示分类的准确性
2. **recall** 召回率: ==检索到的==项目数量占应检索总数的百分比
3. **F1-score**: 好的分类器应该有高精度和高召回率，但在实际应用中两者需要权衡，F1分数便是两者的调和平均数

### Ensemble Learning 集成学习
涉及多个模型并把它们组合起来，产生的效果比单一模型更好，可以是分类器、回归器或其他类型的模型

*集成学习的一种特殊类型: 随机森林*

#### Random Forests
个体模型通过决策树构筑(随机)，优点是不会过拟合。它通过分裂节点选择最佳阈值并减少熵，==会使bias(偏差)增加但降低variance(方差)，因此获得一个稳健的模型==

#### Extremely Random forests
==不仅随机选择特征子集，阈值也随机选择==，进一步减少了方差来获得稳健结果，决策边界更加光滑

### Confidence measure 置信度度量

### Grid Search 网格搜索
允许我们指定一个值的范围内自动运行配置，以这种实验性的尝试来找出最佳配置

### Relative Feature Importance 相对特征重要性
在处理N维数据点的数据集时，有些特征更具判别性，我们可以以此来降低维度，减少算法复杂性
**AdaBoost** 自适应提升: 分类器能专注于更困难的数据点，之前错误的数据点更有可能加入下一次训练的数据集

## Workshop5: Unsupervised Learning
不使用带有标签的训练数据来构建机器学习模型

### K-Means algorithm K均值算法进行聚类
*聚类(Clustering): 分析数据并发现簇(clusters)*
1. 指定簇类数量，利用数据属性将数据分割成K个子组
2. 随机放置**centroids(质心)**，算法将不同的质心远离，并分配数据点到最近的质心
3. ==在迭代中不断更新新聚类中心==，即重新计算质心位置

### Mean Shift algorithm 均值漂移
常用于**聚类**
他是non-parametric(非参数的)，对==潜在的分布不做假设==
整个特征空间被视为一个概率密度函数，聚类对应局部最大值，K个聚类对应K个峰值
1. 对每个数据点在其周围定义一个窗口
2. 计算窗口的质心
3. 将数据点更新到质心位置
4. 重复，不断靠近聚类的峰值
==**质心(即均值)**不断朝峰值移动==

### Silhouette Scores 轮廓分数
现实世界中的数据难以自然被组织成簇，因此我们利用**轮廓**来检查聚类一致性
==轮廓给出了数据点与聚类的拟合程度的估计值，轮廓分数是衡量数据点与自身聚类的相似程度==
- 取值范围在-1到1之间，越靠近1表示数据点与聚类中的其他数据点相似程度大，越靠近-1则不相似(存在噪声)

### Gaussian Mixture Models 高斯混合模型
*混合模型: 一种概率密度模型，数据由不同成分的分布控制*
1. 半参数化，部分依赖于一组预先定义的函数，用于建模数据的分布时提高更高精度和灵活性，同时平滑数据稀疏导致的差距。
2. 函数定义后，混合模型变为参数化 ***分布被识别，即高斯分布混合模型为参数化模型***


## Workshop6: KNN for building a recommendational system
1. Extracting the nearest neighbors
- 从给定的数据集中找到与输入点最近的数据点
2. Build a K-nearest neighbors classifier
- 使用KNN分类器对数据点进行分类
- 在数据集中找出K个最近的数据点，确定其类别，经过投票后进行分类
3. Computing the similarity scores
- **The Eculidean score**: 使用欧几里得**距离**来计算评分，但它可能是无界的。因此我们将其转化为欧几里得**分数**并使其==存在于0和1间==，欧几里得距离越大，分数越低，对象越不相似
- **The Pearson score**: 利用数据点间的协方差和标准偏差来计算分数，==范围为-1到1，1表示相似，0表示无关，-1表示不同==

### collaborative filtering 协同过滤
识别数据集中对象间模式的过程，以便对新对象做出决策

## Workshop7: Logic Programming 逻辑编程
*kanren 包: 让Python支持逻辑编程*
### 什么是LP?
- 逻辑编程是一种programming paradigm 编程范例，或者说一种编程的方法
- 编程范例：计算机程序通过代码解决问题的方式
- ==命令式、函数式、声明式编程、面向对象、过程式、符号式、逻辑==
- 逻辑的结构：**事实，规则，问题**
### 使用LP
- 通过**Facts 事实**和**Rules 规则**寻找解决方案
- 规则：Logical statements 逻辑陈述

### 数学表达式的匹配
逻辑编程是一种==比较表达式并找出未知值==的有效方式

### 验证素数

### 解析家族树
计算机程序必须指定Goal 目标，当逻辑程序和目标不包含变量时，求解器会生成树，构成用于解决问题并达到目标的搜索空间

## Workshop8: Heuristic Search Techniques 启发式搜索技术
在Solution space 解空间中搜索以得出答案，通过引导搜索算法的启发式方法来进行

### What is Heuristic Search
对于问题，我们利用经验尽可能排除明显错的选项来缩小搜索范围。这种经验法则就是heurstic 启发法。使用启发法来引导搜索的方法就是启发式搜索

### Uninformed versus and Informed Search 盲目搜索与启发式搜索
Informed Search: ==有信息搜索==，利用先验信息或规则来消除不必要路径
Uninformed Search: 不将目标考虑在内
**启发式搜索不能总是找到Most optimal solution最优解**

### Constraint Satisfaction Problems(CSPs) 约束满足问题
*例如数独，下棋等守规则约束的问题*
CSPs是数学问题，变量状态必须遵守约束条件

### Local Search Techniques 本地(局部)搜索
局部搜索是一种满足CSPs的方法。他会不断优化方案，更近目标，因此称为==局部==
计算约束条件和更新与目标的距离的过程是**cost 成本**，局部搜索的目标是每一步都找到最小成本的更新

#### Hill Climbing Search Techniques 

#### Simulated Annealing Search Techniques

### Greedy Search 贪心算法
在每个阶段做出局部最优解以找到全局最优解，优点是能在合理时间内产生近似解与全局最优解接近

## Workshop9: Build AI games

### 在人工智能游戏中使用搜索算法
游戏--搜索树，状态--节点与子节点

### Combinatorial Search 组合搜索
搜索算法的缺点：==brute-force search 暴力搜索(exhaustive search)==
组合搜索：利用启发式方法有效探索空间，减小搜索规模

### The MiniMax Algorithm 
- 极大极小算法用过策略实现目标，他试图==最小化对手试图最大化==的函数

### Alpha-Beta Pruning 阿尔法贝塔叶剪枝
- 比极大极小更智能的算法，能避免搜索不包含解决方案的部份树，即Pruning.
- Alpha: 可能解的==最大下限==
- Beta: 可能解的==最小上限==

### The Negamax Algorithm 尼格拉克斯算法
- 一种极大极小算法的变体，尝试能让对手获得的收益最小

## Workshop10: Artificial Nerual Networks 人工神经网络

### Revisit
- 神经网络旨在模拟人类大脑的学习过程
- 识别数据的潜在模式并进行学习
- 可用于分类，回归，分割

### Training a neural network 训练神经网络
- 处理N-dimensional input data n维的输入数据， input layer: N neurons 输入层：N个神经元
- M distinct classes in training data 训练数据中有M个不同类别， output layer: M neurons 输出层：M个神经元
- 输入层和输出层间的是hidden layers隐藏层

神经网络的操作过程：
1. 收集训练数据并标注
2. 每个神经元会自行像函数一样训练直到==error 误差==低于某个阈值

### Building a Perceptron-based classifier 构造一个基于感知机的分类器
Perceptron 感知机：take inputs, perform computation, produces output 接受输入，执行计算最后产生输出
Constant: 偏差，用于在计算时被加入来产生输出

### Single-layer Neural Network(SLNN)，单层神经网络
### Multi-layer Neural Network 多层神经网络
### Recurrent Neural Networks 循环神经网络

## Workshop11: Deep Learning with Convolutional Neural Networks(CNNS) 卷积神经网络的深度学习
普通神经网络的主要缺点之一是忽略了输入数据的结构，所有数据在输入网络前被转换为单维数组，处理图像能力不足，我们构造a neural network with many layers, It is called deep neural network 深度神经网络
处理深度神经网络的过程：深度学习
卷积神经网络：处理图像时将图像的2D strcutres 2d结构考虑在内

### Architecture of CNNS 卷积神经网络的架构
普通神经网络运行的过程：
1. 输入数据转换为single vector 单个向量
2. 穿过神经网络各层，==每个神经元都与前一层的所有神经元相连(fully connectivity)，每一层内的神经元并不互相连接==，
3. 在尝试处理图像时，weight 权重会快速增加，导致大量参数需要被调整，耗时耗力
卷积网络运行过程：
1. 处理数据时明确图像结构
2. 神经元在三维中排列：width, height, depth，每个神经元只与前一层的一小部分区域相连接
3. 深入各层后不断提取高级特征
- 单个神经元仅对visual cortex 感受野中的刺激作出反应
- CNNS利用==filter 滤波器==原理覆盖在输入图像上，利用多个滤波器提取特征，而非单个神经元与每个前一层的神经元相连接
- CNN架构在单个数据集上表现好，参数减少了且权重重复

### Types of layers in a CNN
- Input layer: 对输入数据==不做处理==
- Convolutional layer: 计算神经元与输入中的patches 补丁 之间的卷积
- Rectified Linear Unit layer 修正线性单元层： 对前一层的输出激活函数，为网络添加非线性
- Pooling layer 池化层：对输出进行采样，保留突出部分 Max pooling: 最大池化
- **Fully Connected layer 全连接层: 计算上层的输出分数**

## Workshop12: Recurrent Neural Networks(RNN)
常用于NLP和NLU(自然语言理解)
activation functions 激活函数：神经元接受输入，到达阈值时输出信号。类似的，达到阈值后神经网络利用激活函数输出
常见的激活函数：
1. Step funciton 阶跃函数
2. Sigmoid function 西格玛函数
3. Tanh function 双曲正切函数
4. ReLU fuction 线性整流函数

### Step function
- 一个简单的函数，超过阈值立刻启动
- 神经网络常使用backpropagetion 反向传播 和gradient descent 梯度下降 来计算不同层的权重，但阶跃函数无法沿着梯度下降的方向推进，也无法更新权重

### Sigmoid function
- 自变量趋于负无穷，函数值趋于0；趋于正无穷，函数值趋于1
- 容易出现梯度消失：函数有陡峭坡度，输入变化大却导致输出变化小

### Tanh function 双曲正切函数
- **输出范围变为-1到1， 数值围绕0中心分布，导数高，梯度高，学习率高**
- 但仍然存在梯度消失问题

### ReLU function 双线整流函数
- 给定负数输入时返回0，给定整数输入时返回该值
- Leaky ReLU: 泄漏式：对于自变量为负数的情况，自定义其斜率，导致所有值都受x影响、

### Architecture of RNNs 卷积神经网络的架构
主要概念：利用序列的先前信息
传统的神经网络中输入输出彼此独立，RNNs中在每一层共享相同参数，减少参数总数，拥有一个hidden state，捕获并存储有关序列的信息
