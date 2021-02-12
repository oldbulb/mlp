# 多层感知机的简单推导以及代码实现

## 多层感知机结构

### **1. 单个神经元**

神经网络中计算的基本单元是神经元，一般称作 **节点(node)**  或者 **单元(unit)** 。节点从其他节点接收输入，或者从外部源接收输入，然后计算输出。每个输入都辅有 **权重** （weight），权重取决于其他输入的相对重要性。节点将函数应用到加权后的输入总和。

<img src="https://pic4.zhimg.com/80/v2-4702cea908ffffc23a0bdcb26135ebeb_720w.png" alt="img" style="zoom:50%;" />

此网络接受$ X_1$ 和 $X_2$的数值输入，其权重分别为 $w_1$ 和 $ w_2$。另外，还有配有偏置 $b$的输入。偏置的主要功能是为每一个节点提供可训练的常量值。神经元的输出 Y 如图所示进行计算。

函数 $f$ 是非线性的，叫做**激活函数**。

**激活函数**的作用是将非线性引入神经元的输出。

---

* **sigmoid 函数 **

sigmoid函数也叫Logistic函数，用于隐藏层的输出，输出在(0,1)之间，可以用来做二分类。

公式如下：
$$
\sigma(x)=\frac{1}{1+e^{-x}} \\
{\sigma}^{\prime}(x)=\sigma(x)(1-\sigma(x))
$$


<img src="https://pic2.zhimg.com/80/v2-83469109cd362f5fcf1decf109007fbd_720w.png" alt="img" style="zoom: 50%;" />


  * 缺点：

    * 梯度消失
    * 输出不是zero-centered
    * 计算成本高昂，反向传播求导涉及除法

---

* **tanh 函数**

 tanh 即 Hyperbolic Tangent

 公式如下：
$$
\tanh (x) =  \frac{e^x - e^{-x}}{e^x+e^{-x}} \\
{\tanh}^{\prime}(x)=1-{\tanh}^{2}(x)
$$

<img src="https://pic2.zhimg.com/80/v2-a39596b282f6333bced6e7bfbfe04dcd_720w.png" alt="img" style="zoom:50%;" />

* 缺点：
   * 梯度消失
   * 输出不是zero-centered
   * 计算成本高昂，反向传播求导涉及除法

---

* **ReLU 函数**

ReLU函数其实就是一个取最大值函数，可以取sub-gradient

公式如下：
$$
\mathrm{Relu}(x)= \max(0,x)
$$

<img src="https://pic3.zhimg.com/80/v2-5c97f377cdb5d1f0bc3faf23423c4952_720w.png" alt="img" style="zoom:50%;" />

有以下几大优点：

* 解决了梯度消失问题
* 计算速度很快
* 收敛速度也很快

同时需要注意

* 输出不是zero-centered

* Dead ReLU Problem的问题，指的是某些神经元可能永远不会被激活，导致相应的参数永远不能被更新。

  两个主要原因：(1) 非常不幸的参数初始化；(2) learning rate太高导致在训练过程中参数更新太大，不幸使网络进入这种状态。

  解决方法：采用Xavier初始化方法，以及避免将learning rate设置太大或使用Adagrad等自动调节learning rate的算法。

---

* **Leaky ReLU 函数**

为了解决Dead ReLU Problem，将ReLU的前半段设为 $0.01x$ 而非0。或是基于参数的方法，即Parametric ReLU。

公式如下：
$$
f(x)=\max(0.01x,x)\\
f(x)=\max(\alpha x,x)
$$

<img src="https://pic1.zhimg.com/80/v2-8fa15614231fd01a659d4763beec9b24_720w.png" alt="img" style="zoom:50%;" />

---

5.  **ELU  函数**

   ELU 即 Exponential Linear Units，有着ReLU的基本优点，并且不会出Dead ReLU Problem，输出的均值也接近于0（zero-centered）

   公式如下：
   $$
   f(x)=\left\{\begin{array}{ll}
   x, & \text { if } x>0 \\
   \alpha\left(e^{x}-1\right), & \text { otherwise }
   \end{array}\right.
   $$
   

---

### **2. 前馈神经网络**

前馈神经网络包含了安排在多个层中的多个神经元（节点）。相邻层的节点有连接或者边（edge）。所有的连接都配有权重。

一个前馈神经网络可以包含三种节点：

1. 输入节点（Input Nodes）：输入节点从外部世界提供信息，总称为「输入层」。在输入节点中，不进行任何的计算——仅向隐藏节点传递信息。
2. 隐藏节点（Hidden Nodes）：隐藏节点和外部世界没有直接联系。这些节点进行计算，并将信息从输入节点传递到输出节点。隐藏节点总称为「隐藏层」。尽管一个前馈神经网络只有一个输入层和一个输出层，但网络里可以没有也可以有多个隐藏层。
3. 输出节点（Output Nodes）：输出节点总称为「输出层」，负责计算，并从网络向外部世界传递信息。



>***反向传播误差***  通常缩写为「Back-Propagation」，是几种训练人工神经网络的方法之一。这是一种监督学习方法，即通过标记的训练数据来学习。简单说来，Back-Propagation 就像「从错误中学习」。
>
>一个人工神经网络包含多层的节点；输入层，中间隐藏层和输出层。相邻层节点的连接都有配有「权重」。学习的目的是为这些边缘分配正确的权重。通过输入向量，这些权重可以决定输出向量。*
>
>在监督学习中，训练集是已标注的。这意味着对于一些给定的输入，我们知道期望 / 期待的输出（标注）。
>
>***反向传播算法***  最初，所有的边权重（edge weight）都是随机分配的。对于所有训练数据集中的输入，人工神经网络都被激活，并且观察其输出。这些输出会和我们已知的、期望的输出进行比较，误差会「传播」回上一层。该误差会被标注，权重也会被相应的「调整」。该流程重复，直到输出误差低于制定的标准。
>
>上述算法结束后，我们就得到了一个学习过的人工神经网络，该网络被认为是可以接受「新」输入的。该人工神经网络可以说从几个样本（标注数据）和其错误（误差传播）中得到了学习。



![image-20210212162144997](/Users/oldbulb/Library/Application Support/typora-user-images/image-20210212162144997.png)



## **前馈神经网络-数学推导**

以单隐藏层MLP为例

| Forward-Propagation            |                                                              |
| ------------------------------ | ------------------------------------------------------------ |
| The input of **Input Layer**   | $\boldsymbol{x}={[x_1, x_2, ...,x_d]}^{\top}\in\mathbb{R^{d\times1}}$ |
| The input of **Hidden Layer**  | $\boldsymbol{\alpha}=\boldsymbol{W}^{(1)}\boldsymbol{x}+\boldsymbol{b}^{(1)} \qquad \boldsymbol{\alpha}\in\mathbb{R^{q\times1}}, \boldsymbol{W}^{(1)}\in \mathbb{R^{q\times d}},\boldsymbol{b}^{(1)}\in\mathbb{R^{q\times1}}, {\alpha}_h=\sum_{i=1}^{d}w_{i,h}^{(1)}x_i$ |
| The output of **Hidden Layer** | $\boldsymbol{h}=\sigma(\boldsymbol{\alpha})=\sigma(\boldsymbol{W}^{(1)}\boldsymbol{x}+\boldsymbol{b}^{(1)}) \qquad \boldsymbol{h}\in\mathbb{R^{q\times1}}, {h}_{h}=\sigma(\alpha_h)$ |
| The input of **Output Layer**  | $\boldsymbol{\beta}=\boldsymbol{W}^{(2)}\boldsymbol{h}+\boldsymbol{b}^{(2)} \qquad \boldsymbol{\alpha}\in\mathbb{R^{c\times1}}, \boldsymbol{W}^{(2)}\in \mathbb{R^{c\times q}},\boldsymbol{c}^{(1)}\in\mathbb{R^{q\times1}}, {\beta}_j=\sum_{h=1}^{c}w_{h,j}^{(2)}h_h$ |
| The output of **Output Layer** | $\boldsymbol{y}=f(\boldsymbol{\beta})=f(\boldsymbol{W}^{(2)}\boldsymbol{h}+\boldsymbol{b}^{(2)}) \qquad \boldsymbol{y}\in \mathbb{R^{c\times1}}, y_i=f({\beta}_i)$ |
| Label Vector                   | $\boldsymbol{l}\in\mathbb{R^{c\times1}}$                     |
| Loss function(MSE)             | $\mathcal{L}(\boldsymbol{y},\boldsymbol{l})=\frac{1}{2}{(\boldsymbol{l}-\boldsymbol{y})}^{\top}{(\boldsymbol{l}-\boldsymbol{y})}$ |
| Loss function(Cross-Entropy)   | ${\mathcal{L}(\boldsymbol{y},\boldsymbol{l})=-{\boldsymbol{l}}^{\top}}\log(\boldsymbol{y})$ |
| **Forward-Propagation**        |                                                              |
| Gradient of **Output Layer**   | $\boldsymbol{g}=\frac{\partial \mathcal{L}}{\partial \boldsymbol{\beta}}=\frac{\partial \mathcal{L}}{\partial \boldsymbol{y}}\odot f^{\prime}(\boldsymbol{\beta})$<br />$\mathrm{d}\mathcal{L}=\mathrm{tr}({\frac{\partial \mathcal{L}}{\partial \boldsymbol{\beta}}}^{\top}\cdot\mathrm{d}{\boldsymbol{\beta}})=\mathrm{tr}({\boldsymbol{g}}^{\top}\cdot\mathrm{d}(\boldsymbol{W}^{(2)}\boldsymbol{h}+\boldsymbol{b}^{(2)}))$<br />$\mathrm{d}\mathcal{L}=\mathrm{tr}({\boldsymbol{g}}^{\top}\cdot\mathrm{d}\boldsymbol{W}^{(2)}\cdot\boldsymbol{h})+\mathrm{tr}({\boldsymbol{g}}^{\top}\cdot\boldsymbol{W}^{(2)}\cdot\mathrm{d}\boldsymbol{h})+\mathrm{tr}(\boldsymbol{g}^{\top}\cdot \mathrm{d}{\boldsymbol{b}^{(2)}})$ |
| Gradient of **Parameters**     | $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\boldsymbol{W}}^{(2)}}=\boldsymbol{g}{\boldsymbol{h}}^{\top} \iff \frac{\partial \mathcal{L}}{\partial {w}_{h, j}^{(2)}}=g_jh_h$<br />$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\boldsymbol{b}}^{(2)}}=\boldsymbol{g} \iff \frac{\partial \mathcal{L}}{\partial {b}_{j}^{(2)}}=g_j$<br />$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\boldsymbol{h}}}={{\boldsymbol{W}}^{(2)}}^{\top}\frac{\partial \mathcal{L}}{\partial \boldsymbol{\beta}}={{\boldsymbol{W}}^{(2)}}^{\top}\boldsymbol{g}$ |
| Gradient of **Hidden Layer**   | $\boldsymbol{e}=\frac{\partial \mathcal{L}}{\partial \boldsymbol{\alpha}}=\frac{\partial \mathcal{L}}{\partial \boldsymbol{h}}\odot {\sigma}^{\prime}(\boldsymbol{\alpha})={{\boldsymbol{W}}^{(2)}}^{\top}\boldsymbol{g} \odot {\sigma}^{\prime}(\boldsymbol{\alpha})$<br />$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\boldsymbol{W}}^{(1)}}=\boldsymbol{e}{\boldsymbol{x}}^{\top} \iff \frac{\partial \mathcal{L}}{\partial {w}_{i,h}^{(1)}}={e}_{h}{x_i}^{\top}$<br />$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\boldsymbol{b}}^{(1)}}=\boldsymbol{e} \iff \frac{\partial \mathcal{L}}{\partial {b}_{h}^{(1)}}={e}_{h}$ |
| Learning Rate                  | $\eta$                                                       |
| Parameters update              | $\Delta{\boldsymbol{W}}^{(2)}=-\eta \boldsymbol{g}{\boldsymbol{h}}^{\top} \qquad \Delta{\boldsymbol{b}}^{(2)}=-\eta \boldsymbol{g}$<br />$\Delta{\boldsymbol{W}}^{(1)}=-\eta \boldsymbol{e}{\boldsymbol{x}}^{\top} \qquad \Delta{\boldsymbol{b}}^{(1)}=-\eta \boldsymbol{e}$<br />$\theta \leftarrow \theta + \Delta\theta$ |



## 矩阵求导





## **梯度下降**





## **Basic Algorithms** **&** **Strategies**

**Stochastic Gradient Descent** 

**Momentum &** **Nesterov Momentum**

**Parameter Initialization Strategies**

**Algorithms with Adaptive Learning Rates** 

 **AdaGrad**

 **RMSProp**

 **Adam**





