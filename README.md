# 3DPointCloud-Notes
This is the collection of my notes on the online course 3D pointcloud processing in Shenlanxueyuan.

## Content
* [1. Introduction](#1-introduction)
* [2. Nearest Neighbor](#2-nearest-neighbor)
* [3. Clustering](#3-clustering)
* [4. Model Fitting](#4-model-fitting)
* [5. Deep Learning](#5-deep-learning)
* [6. Object Detection](#6-object-detection)
* [7. Feature Detection](#7-feature-dtection)
* [8. Feature Description](#8-feature-description)


## 1. Introduction
   * **3D point Cloud**
        * Representation: N x (3+D) 
        * Data Source:
            * Lidar
            * RGB-D
            * CAD
   * 3D description
        * mesh
        * Voxel grid: 不高效，因为稀疏性。
        * octree
   * Difficulties
        * Sparsity (far)
        * Irregular (compared to pixel) in neighbor searching 
        * No texture information  (three people and a car)
        * Un-order (in matrix) - difficult in deep learning
        * Rotation invariance
   * **PCA (linear)** - find the dominant direction
       * Application:
            * Dimensionality reduction
            * Surface normal
            * Classification 
       * SVD
       * Spectrum theorem
       * Rayleigh Quotient 
       * **Theory:**
           * Input: data
           * Output: principle vectors (large variance)
       * Process:
           * Normalization
           * Calculate variance with projection using matrix form
           * Maximize variance to find z1=u1
           * Find z2 by deflation (remove z1 effect), which leads to z1=u2
           * note:  eigen values should be in order!
   * **Kernel PCA (nonlinear)** - lift dimension 
       * Process:
           * Find extra dimension
           * PCA
       * Problem
           * How to determine phi()
           * How to deal with high dimension
       * 见笔记
   * **Surface Normal**
       * PCA:
           * Normal - least significant vector
           * Curvature - lambda3/(lambda3+lambda2+lambda1)
       * Weighted normal estimation: 利用相似度决定权重
       * Noise removal
           * Radius
           * Statistical 
       * Downsampling
           * Voxel grid
               * Hash table
       * Farthest point sampling (FPS)
       * Normal space sampling (NSS): 
           * 法向量角度考虑均匀分布
           * 用于ICP
       * Upsampling 
           * bilateral filter - gaussian
               * Edge preserving - 利用颜色相似度


## 2. Nearest Neighbor
   * Nearest Neighbor
       * KNN
       * Radius-NN
   * Binary Search Tree - 一维比较大小
       * 1NN search - worst distantce
       * KNN search
   * Kd-Tree - 高维二叉树
       * splitting
           * Round-robin
           * Widest spread
   * Octree
   
   
## 3. Clustering
* **K-Means**
   * Process 
       * Random
       * Average
   * Trick 
       * Better initialization 
   * 容易被噪声干扰
       * K-medoid: V(~) 距离之和
   * Application
       * 图像颜色压缩
   * 问题：
       * k不知道
       * 对噪声敏感
       * Hard assignment，不是概率信息
* **Gaussian Mixture Model (GMM)**
   * MLE
       * Singularity: 方差非常小，或者一个点代表一个分布
           * MAP
           * Bayesian
   * https://zhuanlan.zhihu.com/p/40024110
* **Expectation - Maximization (EM)**
   * 用MLE估算模型参数by adding latent variable Z
   * 证明：https://blog.csdn.net/henryczj/article/details/40786597
   * 最后需要注意的是，如果目标函数不是凸函数，那么EM算法的解将是局部最优解，所以在实际工程中，参数初始值的选取十分重要，常用的办法是选取几个不同的初始值进行迭代，然后对得到的值进行比较，从中选取最好的。（收敛性不等于最优性）
* **Spectral Clustering**
   * 不规则，因为不基于欧式距离，而是connectivity
   * K is auto-determined - eigen gap
   * Process 
       * Graph build - adjacency matrix W
       * degree matrix D
       * Laplace matrix L=D-W 
       * 放松，所以对eigenvector聚类
   * Graph Min-cut
       * Minimize cut edge sum
       * Constraint with size of clustering to prevent degeneration 
           * Size:
               * Size(A)=|A| 点个数 - approx ratiocut
                   * 容易对孤立节点有单独分配                   
                   * 所以对eigenvector聚类但是可以用来降噪？
               * size(A)=vol(A) edge和 - approx normalizedcut （类似？不会与点的数量有关吗？）
* **Mean shift**
   * 有点像gradient
   * 还是假设ellipse
* **DBSCAN** - density based spatial clustering of application with noise
   * 连接附近点   
   
   
## 4. Model Fitting
* **Linear Square**
   * Sensitive to noise
       * Modify loss
           * L1
           * L2
           * Log
           * Huber
       * Nonlinearity
           * Optimization
* **Hough Transform** - 参数空间
   * prefer in Three or less parameter
* **RANSAC** (Random Sample consensus)
   * 很多模型选一个


## 5. Deep Learning
* **CNN**
* **VoxelCNN**
   * 不可拓展
   * 过大
* **MVCNN**
   * 投射到不同二维空间
   * 计算量过大
* **PointNet**
   * 为什么要拉长？
* **PointNet++**
   * 多层特征提取


## 6. Object Detection
* **Object detection**
   * Localization & Classification
   * Precision & Recall
   * Average Precision (AP) - one category 
   * mAP - who network
   * Non-maximum surpression (NMS) - multiple boxes
* **Image based**
   * RCNN
       * Region clustering + classification 
* **VoxelNet & PointPillars**
* **PointRCNN**

   
## 7. Feature Detection
* Basic Idea
   * Flat
   * Edge
   * Corner
* **Harris Corner**
   * u,v is small  —>  gradient
   * Taylor expansion
   * Meaning of derivative
* **Harris 3D**
   * Intensity gradient
       * e=[ex,ey,ez]T
       * 改写成向量形式，得到e=(A^T*A)^-1*A^T*b
       * e投影到local surface！！！！！！！
* 总结
* **Harris 6D**
* ISS (Intrinsic Shape Signatures) 
   * weighted 3d PCA 
   * 3d constraint
* USIP
   * unsupervised learning with probability chamfer loss
   * 寻找旋转平移矩阵 - 不变性
   * 退化
       * 中心点
       * principle axis
   * 限制感知域 - 尺度性
* SO-Net
   * 切割点云，分块处理
   * Self-organizing map
   * grouping
       * Node-to-point
       * Point-to-node ✔️
* 3D feat-net
   
   
## 8. Feature Description   
* Basic Idea
   * Detection - keypoints
   * Description - vector
   * Matching  - correspondence 
* Handcrafted
   * Histogram
       * PFH (point feature histogram)
           * 不关心坐标，如镜像
           * 对法向量敏感
       * FPFH (fast point feature histogram) 
           * Weighted sum of SPFH
           * Concatenated 
       * Signature 
           * SHOT  (signature of histogram of Orientation)
               * 旋转？local reference frame (LRF)？
               * Boundary effect
* DL - 对旋转敏感
   * 3DMatch
       * Positive - negative
       * Contrastive lost
       * triplet loss
   * PerfectMatch
       * LRF
   * PPFnet (point pair feature)
       * feature：pos, norm, SPFH
       * N-tuple loss
           * 矩阵形式的contrastive loss
   * PPF-FoldeNet
       * auto encoder
* 总结
   
   
   
   
