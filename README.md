# 计算广告论文、学习资料、业界分享
在这里动态更新我工作中实现或者阅读过的计算广告相关论文、学习资料和业界分享。作为自己工作的整理和总结，也希望能为计算广告相关行业的技术同学带来便利。所有资料均来自于互联网，如有侵权，请联系[王喆](http://wangzhe.website/about/)

**下面将列出所有的资料目录，以及我对每篇文章的简要介绍** <br>如有任何问题，欢迎对计算广告感兴趣的同学与我讨论，我的联系方式如下：
* email: wzhe06@gmail.com
* LinkedIn: [王喆的LinkedIn](https://www.linkedin.com/in/zhe-wang-profile/)
* 知乎私信: [王喆的知乎](https://www.zhihu.com/people/wang-zhe-58)

**会不断加入一些重要的计算广告相关论文和资料，并去掉一些过时的或者跟计算广告不太相关的论文**

* `New!` [Image Matters- Visually modeling user behaviors using Advanced Model Server.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Image%20Matters-%20Visually%20modeling%20user%20behaviors%20using%20Advanced%20Model%20Server.pdf) <br />
阿里提出引入商品图像特征的（Deep Image CTR Model）CTR预估模型，并介绍其分布式机器学习框架 Advanced Model Server (AMS)
* `New!` [Deep Interest Network for Click-Through Rate Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Deep%20Interest%20Network%20for%20Click-Through%20Rate%20Prediction.pdf) <br />
阿里提出的深度兴趣网络（Deep Interest Network）CTR预估模型


## 目录

### Optimization Method
Online Optimization，Parallel SGD，FTRL等优化方法，实用并且能够给出直观解释的文章
* [Google Vizier A Service for Black-Box Optimization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Google%20Vizier%20A%20Service%20for%20Black-Box%20Optimization.pdf) <br />
* [在线最优化求解(Online Optimization)-冯扬.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/%E5%9C%A8%E7%BA%BF%E6%9C%80%E4%BC%98%E5%8C%96%E6%B1%82%E8%A7%A3%28Online%20Optimization%29-%E5%86%AF%E6%89%AC.pdf) <br />
非常推荐冯扬的这个教程，把在线优化问题讲的非常透
* [Hogwild A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Hogwild%20A%20Lock-Free%20Approach%20to%20Parallelizing%20Stochastic%20Gradient%20Descent.pdf) <br />
* [Parallelized Stochastic Gradient Descent.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Parallelized%20Stochastic%20Gradient%20Descent.pdf) <br />
* [A Survey on Algorithms of the Regularized Convex Optimization Problem.pptx](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/A%20Survey%20on%20Algorithms%20of%20the%20Regularized%20Convex%20Optimization%20Problem.pptx) <br />
* [Follow-the-Regularized-Leader and Mirror Descent- Equivalence Theorems and L1 Regularization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Follow-the-Regularized-Leader%20and%20Mirror%20Descent-%20Equivalence%20Theorems%20and%20L1%20Regularization.pdf) <br />
* [A Review of Bayesian Optimization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/A%20Review%20of%20Bayesian%20Optimization.pdf) <br />
* [Taking the Human Out of the Loop- A Review of Bayesian Optimization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Taking%20the%20Human%20Out%20of%20the%20Loop-%20A%20Review%20of%20Bayesian%20Optimization.pdf) <br />
* [非线性规划.doc](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/%E9%9D%9E%E7%BA%BF%E6%80%A7%E8%A7%84%E5%88%92.doc) <br />


### CTR Prediction
CTR预估模型相关问题，作为计算广告的核心，CTR预估永远是研究的热点，下面每一篇都是非常流行的文章，推荐逐一精读
* [Deep Crossing- Web-Scale Modeling without Manually Crafted Combinatorial Features.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Deep%20Crossing-%20Web-Scale%20Modeling%20without%20Manually%20Crafted%20Combinatorial%20Features.pdf) <br />
* [Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Learning%20Piece-wise%20Linear%20Models%20from%20Large%20Scale%20Data%20for%20Ad%20Click%20Prediction.pdf) <br />
阿里提出的Large Scale Piece-wise Linear Model (LS-PLM) CTR预估模型
* [[FNN]Deep Learning over Multi-field Categorical Data.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/%5BFNN%5DDeep%20Learning%20over%20Multi-field%20Categorical%20Data.pdf) <br />
* [Entire Space Multi-Task Model_ An Effective Approach for Estimating Post-Click Conversion Rate.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Entire%20Space%20Multi-Task%20Model_%20An%20Effective%20Approach%20for%20Estimating%20Post-Click%20Conversion%20Rate.pdf) <br />
* [Deep Interest Network for Click-Through Rate Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Deep%20Interest%20Network%20for%20Click-Through%20Rate%20Prediction.pdf) <br />
* [Product-based Neural Networks for User Response Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Product-based%20Neural%20Networks%20for%20User%20Response%20Prediction.pdf) <br />
张伟楠博士的另外一篇论文，提出了 PNN 模型，在 FNN 基础上对特征的隐向量进行了 inner product 作为新特征
* [Bid-aware Gradient Descent for Unbiased Learning with Censored Data in Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Bid-aware%20Gradient%20Descent%20for%20Unbiased%20Learning%20with%20Censored%20Data%20in%20Display%20Advertising.pdf) <br />
RTB 中训练 CTR 模型数据集是赢得出价的广告，预测时的样本却是所有候选的广告，也就是训练集和测试集的分布不一致，这篇文章就是要消除这样的 bias
* [Ad Click Prediction a View from the Trenches.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Ad%20Click%20Prediction%20a%20View%20from%20the%20Trenches.pdf) <br />
Google大名鼎鼎的用FTRL解决CTR在线预估的工程文章，非常经典。
* [Image Matters- Visually modeling user behaviors using Advanced Model Server.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Image%20Matters-%20Visually%20modeling%20user%20behaviors%20using%20Advanced%20Model%20Server.pdf) <br />
阿里提出引入商品图像特征的（Deep Image CTR Model）CTR预估模型，并介绍其分布式机器学习框架 Advanced Model Server (AMS)
* [[DeepFM]- A Factorization-Machine based Neural Network for CTR Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/%5BDeepFM%5D-%20A%20Factorization-Machine%20based%20Neural%20Network%20for%20CTR%20Prediction.pdf) <br />
* [Logistic Regression in Rare Events Data.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Logistic%20Regression%20in%20Rare%20Events%20Data.pdf) <br />
样本稀少情况下的LR模型训练，讲的比较细
* [Deep & Cross Network for Ad Click Predictions.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Deep%20%26%20Cross%20Network%20for%20Ad%20Click%20Predictions.pdf) <br />
Google 在17年发表的 Deep&Cross 网络，类似于 Wide&Deep, 比起 PNN 只做了特征二阶交叉，Deep&Cross 理论上能够做任意高阶的特征交叉
* [An Overview of Multi-Task Learning in Deep Neural Networks.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/An%20Overview%20of%20Multi-Task%20Learning%20in%20Deep%20Neural%20Networks.pdf) <br />
* [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Learning%20Deep%20Structured%20Semantic%20Models%20for%20Web%20Search%20using%20Clickthrough%20Data.pdf) <br />
* [Wide & Deep Learning for Recommender Systems.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Wide%20%26%20Deep%20Learning%20for%20Recommender%20Systems.pdf) <br />
Google 的 Wide & Deep 模型，论文将模型用于推荐系统中，但也可用于 CTR 预估中
* [Adaptive Targeting for Online Advertisement.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Adaptive%20Targeting%20for%20Online%20Advertisement.pdf) <br />
一篇比较简单但是全面的CTR预估的文章，有一定实用性
* [Practical Lessons from Predicting Clicks on Ads at Facebook.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Practical%20Lessons%20from%20Predicting%20Clicks%20on%20Ads%20at%20Facebook.pdf) <br />
Facebook的一篇非常出名的文章，GBDT＋LR/FM解决CTR预估问题，工程性很强

### Topic Model
话题模型相关文章，PLSA，LDA，进行广告Context特征提取，创意优化经常会用到Topic Model
* [概率语言模型及其变形系列.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/%E6%A6%82%E7%8E%87%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E5%8F%98%E5%BD%A2%E7%B3%BB%E5%88%97.pdf) <br />
* [Parameter estimation for text analysis.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/Parameter%20estimation%20for%20text%20analysis.pdf) <br />
* [LDA数学八卦.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/LDA%E6%95%B0%E5%AD%A6%E5%85%AB%E5%8D%A6.pdf) <br />
* [Distributed Representations of Words and Phrases and their Compositionality.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/Distributed%20Representations%20of%20Words%20and%20Phrases%20and%20their%20Compositionality.pdf) <br />
* [Dirichlet Distribution, Dirichlet Process and Dirichlet Process Mixture(PPT).pdf](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/Dirichlet%20Distribution%2C%20Dirichlet%20Process%20and%20Dirichlet%20Process%20Mixture%28PPT%29.pdf) <br />
* [理解共轭先验.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/%E7%90%86%E8%A7%A3%E5%85%B1%E8%BD%AD%E5%85%88%E9%AA%8C.pdf) <br />


### Google Three Papers
Google三大篇，HDFS，MapReduce，BigTable，奠定大数据基础架构的三篇文章，任何从事大数据行业的工程师都应该了解
* [MapReduce Simplified Data Processing on Large Clusters.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Google%20Three%20Papers/MapReduce%20Simplified%20Data%20Processing%20on%20Large%20Clusters.pdf) <br />
* [The Google File System.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Google%20Three%20Papers/The%20Google%20File%20System.pdf) <br />
* [Bigtable A Distributed Storage System for Structured Data.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Google%20Three%20Papers/Bigtable%20A%20Distributed%20Storage%20System%20for%20Structured%20Data.pdf) <br />


### Factorization Machines
FM因子分解机模型的相关paper，在计算广告领域非常实用的模型
* [FM PPT by CMU.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/FM%20PPT%20by%20CMU.pdf) <br />
* [Field-aware Factorization Machines for CTR Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/Field-aware%20Factorization%20Machines%20for%20CTR%20Prediction.pdf) <br />
* [Factorization Machines Rendle2010.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/Factorization%20Machines%20Rendle2010.pdf) <br />
* [libfm-1.42.manual.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/libfm-1.42.manual.pdf) <br />
* [Scaling Factorization Machines to Relational Data.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/Scaling%20Factorization%20Machines%20to%20Relational%20Data.pdf) <br />
* [Fast Context-aware Recommendations with Factorization Machines.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/Fast%20Context-aware%20Recommendations%20with%20Factorization%20Machines.pdf) <br />
* [fastFM- A Library for Factorization Machines.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/fastFM-%20A%20Library%20for%20Factorization%20Machines.pdf) <br />


### Budget Control
广告系统中Pacing，预算控制，以及怎么把预算控制与其他模块相结合的问题
* [Budget Pacing for Targeted Online Advertisements at LinkedIn.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Budget%20Pacing%20for%20Targeted%20Online%20Advertisements%20at%20LinkedIn.pdf) <br />
linkedin的一篇非常有工程价值的解决pacing问题的文章，强烈建议计算广告系统采用此方法。
* [Predicting Traffic of Online Advertising in Real-time Bidding Systems from Perspective of Demand-Side Platforms.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Predicting%20Traffic%20of%20Online%20Advertising%20in%20Real-time%20Bidding%20Systems%20from%20Perspective%20of%20Demand-Side%20Platforms.pdf) <br />
* [Real Time Bid Optimization with Smooth Budget Delivery in Online Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Real%20Time%20Bid%20Optimization%20with%20Smooth%20Budget%20Delivery%20in%20Online%20Advertising.pdf) <br />
如何将Pcaing与效果优化结合在一起，这篇文章讲的很清楚
* [PID控制经典培训教程.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/PID%E6%8E%A7%E5%88%B6%E7%BB%8F%E5%85%B8%E5%9F%B9%E8%AE%AD%E6%95%99%E7%A8%8B.pdf) <br />
PID控制的经典教程
* [PID控制原理与控制算法.doc](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/PID%E6%8E%A7%E5%88%B6%E5%8E%9F%E7%90%86%E4%B8%8E%E6%8E%A7%E5%88%B6%E7%AE%97%E6%B3%95.doc) <br />
对于采用PID控制解决pacing问题，该文章是PID控制原理比较清晰的介绍文章。
* [Smart Pacing for Effective Online Ad Campaign Optimization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Smart%20Pacing%20for%20Effective%20Online%20Ad%20Campaign%20Optimization.pdf) <br />
跟上篇文章一样，都是雅虎同一组人写的，解决预算控制与效果结合的问题，可以跟上篇文章一起看了

### Tree Model
树模型和基于树模型的boosting模型，树模型的效果在大部分问题上非常好，在CTR，CVR预估及特征工程方面的应用非常广
* [Introduction to Boosted Trees.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Introduction%20to%20Boosted%20Trees.pdf) <br />
* [Classification and Regression Trees.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Classification%20and%20Regression%20Trees.pdf) <br />
* [Greedy Function Approximation A Gradient Boosting Machine.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Greedy%20Function%20Approximation%20A%20Gradient%20Boosting%20Machine.pdf) <br />
* [Classification and Regression Trees.ppt](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Classification%20and%20Regression%20Trees.ppt) <br />


### Guaranteed Contracts Ads
事实上，现在很多大的媒体主仍是合约广告系统，合约广告系统的在线分配，Yield Optimization，以及定价问题都是非常重要且有挑战性的问题
* [A Dynamic Pricing Model for Unifying Programmatic Guarantee and Real-Time Bidding in Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/A%20Dynamic%20Pricing%20Model%20for%20Unifying%20Programmatic%20Guarantee%20and%20Real-Time%20Bidding%20in%20Display%20Advertising.pdf) <br />
* [Pricing Guaranteed Contracts in Online Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/Pricing%20Guaranteed%20Contracts%20in%20Online%20Display%20Advertising.pdf) <br />
* [Risk-Aware Dynamic Reserve Prices of Programmatic Guarantee in Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/Risk-Aware%20Dynamic%20Reserve%20Prices%20of%20Programmatic%20Guarantee%20in%20Display%20Advertising.pdf) <br />
* [Pricing Guidance in Ad Sale Negotiations The PrintAds Example.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/Pricing%20Guidance%20in%20Ad%20Sale%20Negotiations%20The%20PrintAds%20Example.pdf) <br />
* [Risk-Aware Revenue Maximization in Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/Risk-Aware%20Revenue%20Maximization%20in%20Display%20Advertising.pdf) <br />


### Bidding Strategy
计算广告中广告定价，RTB过程中广告出价策略的相关问题
* [Research Frontier of Real-Time Bidding based Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Research%20Frontier%20of%20Real-Time%20Bidding%20based%20Display%20Advertising.pdf) <br />
张伟楠博士的一篇介绍竞价算法的ppt，可以非常清晰的了解该问题的主要方法
* [Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Budget%20Constrained%20Bidding%20by%20Model-free%20Reinforcement%20Learning%20in%20Display%20Advertising.pdf) <br />
* [Real-Time Bidding with Multi-Agent Reinforcement Learning in Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Real-Time%20Bidding%20with%20Multi-Agent%20Reinforcement%20Learning%20in%20Display%20Advertising.pdf) <br />
* [Real-Time Bidding by Reinforcement Learning in Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Real-Time%20Bidding%20by%20Reinforcement%20Learning%20in%20Display%20Advertising.pdf) <br />
* [Combining Powers of Two Predictors in Optimizing Real-Time Bidding Strategy under Constrained Budget.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Combining%20Powers%20of%20Two%20Predictors%20in%20Optimizing%20Real-Time%20Bidding%20Strategy%20under%20Constrained%20Budget.pdf) <br />
国立台湾大学的文章，介绍一种基于流量选择的计算广告竞价方法，有别于传统的CTR CPC的方法，我在实践中尝试过该方法，非常有效
* [Real-Time Bidding Algorithms for Performance-Based Display Ad Allocation.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Real-Time%20Bidding%20Algorithms%20for%20Performance-Based%20Display%20Ad%20Allocation.pdf) <br />
微软的一篇基于PID反馈控制的与效果相关的竞价算法
* [Deep Reinforcement Learning for Sponsored Search Real-time Bidding.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Deep%20Reinforcement%20Learning%20for%20Sponsored%20Search%20Real-time%20Bidding.pdf) <br />
阿里妈妈搜索广告团队的论文，通过 Reinforcement Learning 探索实时出价问题<br />

### Computational Advertising Architect
广告系统的架构问题
* [Parameter Server for Distributed Machine Learning.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Parameter%20Server%20for%20Distributed%20Machine%20Learning.pdf) <br />
* [[TensorFlow Whitepaper]TensorFlow- Large-Scale Machine Learning on Heterogeneous Distributed Systems.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%5BTensorFlow%20Whitepaper%5DTensorFlow-%20Large-Scale%20Machine%20Learning%20on%20Heterogeneous%20Distributed%20Systems.pdf) <br />
* [大数据下的广告排序技术及实践.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%E5%A4%A7%E6%95%B0%E6%8D%AE%E4%B8%8B%E7%9A%84%E5%B9%BF%E5%91%8A%E6%8E%92%E5%BA%8F%E6%8A%80%E6%9C%AF%E5%8F%8A%E5%AE%9E%E8%B7%B5.pdf) <br />
阿里妈妈的一篇广告排序问题的ppt，模型、训练、评估都有涉及，很有工程价值
* [美团机器学习 吃喝玩乐中的算法问题.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%E7%BE%8E%E5%9B%A2%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%20%E5%90%83%E5%96%9D%E7%8E%A9%E4%B9%90%E4%B8%AD%E7%9A%84%E7%AE%97%E6%B3%95%E9%97%AE%E9%A2%98.pdf) <br />
美团王栋博士的一篇关于美团机器学习相关问题的介绍，介绍的比较全但比较粗浅，可以借此了解美团的一些机器学习问题
* [Display Advertising with Real-Time Bidding (RTB) and Behavioural Targeting.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Display%20Advertising%20with%20Real-Time%20Bidding%20%28RTB%29%20and%20Behavioural%20Targeting.pdf) <br />
张伟楠博士的RTB过程所有相关算法的书，全而精，非常棒
* [A Comparison of Distributed Machine Learning Platforms.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/A%20Comparison%20of%20Distributed%20Machine%20Learning%20Platforms.pdf) <br />
* [Efficient Query Evaluation using a Two-Level Retrieval Process.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Efficient%20Query%20Evaluation%20using%20a%20Two-Level%20Retrieval%20Process.pdf) <br />
搜索广告中经典的搜索算法 Wand(Weak AND)
* [[TensorFlow Whitepaper]TensorFlow- A System for Large-Scale Machine Learning.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%5BTensorFlow%20Whitepaper%5DTensorFlow-%20A%20System%20for%20Large-Scale%20Machine%20Learning.pdf) <br />
* [Scaling Distributed Machine Learning with the Parameter Server.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Scaling%20Distributed%20Machine%20Learning%20with%20the%20Parameter%20Server.pdf) <br />
* [Overlapping Experiment Infrastructure More, Better, Faster Experimentation.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Overlapping%20Experiment%20Infrastructure%20More%2C%20Better%2C%20Faster%20Experimentation.pdf) <br />
Google 一篇关于 A/B 测试框架的论文，涉及到如何切分流量以同时进行多个 A/B 测试，工程性很强

### Machine Learning Tutorial
机器学习方面一些非常实用的学习资料
* [各种回归的概念学习.doc](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E5%90%84%E7%A7%8D%E5%9B%9E%E5%BD%92%E7%9A%84%E6%A6%82%E5%BF%B5%E5%AD%A6%E4%B9%A0.doc) <br />
* [机器学习总图.jpg](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%80%BB%E5%9B%BE.jpg) <br />
* [Efficient Estimation of Word Representations in Vector Space.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space.pdf) <br />
* [Rules of Machine Learning- Best Practices for ML Engineering.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/Rules%20of%20Machine%20Learning-%20Best%20Practices%20for%20ML%20Engineering.pdf) <br />
* [An introduction to ROC analysis.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/An%20introduction%20to%20ROC%20analysis.pdf) <br />
* [Deep Learning Tutorial.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/Deep%20Learning%20Tutorial.pdf) <br />
* [广义线性模型.ppt](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E5%B9%BF%E4%B9%89%E7%BA%BF%E6%80%A7%E6%A8%A1%E5%9E%8B.ppt) <br />
* [贝叶斯统计学(PPT).pdf](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E8%B4%9D%E5%8F%B6%E6%96%AF%E7%BB%9F%E8%AE%A1%E5%AD%A6%28PPT%29.pdf) <br />
* [关联规则基本算法及其应用.doc](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E5%85%B3%E8%81%94%E8%A7%84%E5%88%99%E5%9F%BA%E6%9C%AC%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E5%BA%94%E7%94%A8.doc) <br />


### Transfer Learning
迁移学习相关文章，计算广告中经常遇到新广告冷启动的问题，利用迁移学习能较好解决该问题
* [Scalable Hands-Free Transfer Learning for Online Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Transfer%20Learning/Scalable%20Hands-Free%20Transfer%20Learning%20for%20Online%20Advertising.pdf) <br />
* [A Survey on Transfer Learning.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Transfer%20Learning/A%20Survey%20on%20Transfer%20Learning.pdf) <br />


### Exploration and Exploitation
探索和利用，计算广告中非常经典，也是容易被大家忽视的问题，其实所有的广告系统都面临如何解决新广告主冷启动，以及在效果不好的情况下如何探索新的优质流量的问题，希望该目录下的几篇文章能够帮助到你
* [An Empirical Evaluation of Thompson Sampling.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/An%20Empirical%20Evaluation%20of%20Thompson%20Sampling.pdf) <br />
* [Dynamic Online Pricing with Incomplete Information Using Multi-Armed Bandit Experiments.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Dynamic%20Online%20Pricing%20with%20Incomplete%20Information%20Using%20Multi-Armed%20Bandit%20Experiments.pdf) <br />
* [Finite-time Analysis of the Multiarmed Bandit Problem.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Finite-time%20Analysis%20of%20the%20Multiarmed%20Bandit%20Problem.pdf) <br />
* [A Fast and Simple Algorithm for Contextual Bandits.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/A%20Fast%20and%20Simple%20Algorithm%20for%20Contextual%20Bandits.pdf) <br />
* [Customer Acquisition via Display Advertising Using MultiArmed Bandit Experiments.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Customer%20Acquisition%20via%20Display%20Advertising%20Using%20MultiArmed%20Bandit%20Experiments.pdf) <br />
* [Mastering the game of Go with deep neural networks and tree search.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Mastering%20the%20game%20of%20Go%20with%20deep%20neural%20networks%20and%20tree%20search.pdf) <br />
* [Exploring compact reinforcement-learning representations with linear regression.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Exploring%20compact%20reinforcement-learning%20representations%20with%20linear%20regression.pdf) <br />
* [Incentivizting Exploration in Reinforcement Learning with Deep Predictive Models.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Incentivizting%20Exploration%20in%20Reinforcement%20Learning%20with%20Deep%20Predictive%20Models.pdf) <br />
* [Bandit Algorithms Continued- UCB1.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Bandit%20Algorithms%20Continued-%20UCB1.pdf) <br />
* [A Contextual-Bandit Approach to Personalized News Article Recommendation(LinUCB).pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/A%20Contextual-Bandit%20Approach%20to%20Personalized%20News%20Article%20Recommendation%28LinUCB%29.pdf) <br />
* [Exploitation and Exploration in a Performance based Contextual Advertising System.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Exploitation%20and%20Exploration%20in%20a%20Performance%20based%20Contextual%20Advertising%20System.pdf) <br />
* [Bandit based Monte-Carlo Planning.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Bandit%20based%20Monte-Carlo%20Planning.pdf) <br />
* [Random Forest for the Contextual Bandit Problem.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Random%20Forest%20for%20the%20Contextual%20Bandit%20Problem.pdf) <br />
* [Unifying Count-Based Exploration and Intrinsic Motivation.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Unifying%20Count-Based%20Exploration%20and%20Intrinsic%20Motivation.pdf) <br />
* [Analysis of Thompson Sampling for the Multi-armed Bandit Problem.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Analysis%20of%20Thompson%20Sampling%20for%20the%20Multi-armed%20Bandit%20Problem.pdf) <br />
* [Thompson Sampling PPT.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Thompson%20Sampling%20PPT.pdf) <br />
* [Hierarchical Deep Reinforcement Learning- Integrating Temporal Abstraction and Intrinsic Motivation.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Hierarchical%20Deep%20Reinforcement%20Learning-%20Integrating%20Temporal%20Abstraction%20and%20Intrinsic%20Motivation.pdf) <br />
* [Exploration and Exploitation Problem by Wang Zhe.pptx](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Exploration%20and%20Exploitation%20Problem%20by%20Wang%20Zhe.pptx) <br />
* [Exploration exploitation in Go UCT for Monte-Carlo Go.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Exploration%20exploitation%20in%20Go%20UCT%20for%20Monte-Carlo%20Go.pdf) <br />
* [对抗搜索、多臂老虎机问题、UCB算法.ppt](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/%E5%AF%B9%E6%8A%97%E6%90%9C%E7%B4%A2%E3%80%81%E5%A4%9A%E8%87%82%E8%80%81%E8%99%8E%E6%9C%BA%E9%97%AE%E9%A2%98%E3%80%81UCB%E7%AE%97%E6%B3%95.ppt) <br />
* [Using Confidence Bounds for Exploitation-Exploration Trade-offs.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Using%20Confidence%20Bounds%20for%20Exploitation-Exploration%20Trade-offs.pdf) <br />


### Allocation
广告流量的分配问题
* [An Efficient Algorithm for Allocation of Guaranteed Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Allocation/An%20Efficient%20Algorithm%20for%20Allocation%20of%20Guaranteed%20Display%20Advertising.pdf) <br />
同样是雅虎的流量分配文章，跟上一篇文章同时发布，介绍SHALE流量分配算法
* [Ad Serving Using a Compact Allocation Plan.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Allocation/Ad%20Serving%20Using%20a%20Compact%20Allocation%20Plan.pdf) <br />
雅虎的一篇比较经典的流量分配的文章，文中的HWM和DUAL算法都比较实用
