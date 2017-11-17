# 计算广告论文、学习资料、业界分享
在这里动态更新我工作中实现或者阅读过的计算广告相关论文、学习资料和业界分享。作为自己工作的整理和总结，也希望能为计算广告相关行业的技术同学带来便利。所有资料均来自于互联网，如有侵权，请联系[王喆](http://wangzhe.website/about/)

**下面将列出所有的资料目录，以及我对每篇文章的简要介绍** <br>如有任何问题，欢迎对计算广告感兴趣的同学与我讨论，我的联系方式如下：
* email: wzhe06@163.com
* 知乎私信: [王喆的知乎](https://www.zhihu.com/people/wang-zhe-58)
* 主页留言: [王喆的主页](http://wangzhe.website/about/)

**会不断加入一些重要的计算广告相关论文和资料，并去掉一些过时的或者跟计算广告不太相关的论文**

* `New!` [Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Learning%20Piece-wise%20Linear%20Models%20from%20Large%20Scale%20Data%20for%20Ad%20Click%20Prediction.pdf) <br />
阿里提出的Large Scale Piece-wise Linear Model (LS-PLM) CTR预估模型
* `New!` [Deep Interest Network for Click-Through Rate Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Deep%20Interest%20Network%20for%20Click-Through%20Rate%20Prediction.pdf) <br />
阿里提出的深度兴趣网络（Deep Interest Network）CTR预估模型


## 目录

### Allocation
广告流量的分配问题
* [Ad Serving Using a Compact Allocation Plan.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Allocation/Ad%20Serving%20Using%20a%20Compact%20Allocation%20Plan.pdf) <br />
雅虎的一篇比较经典的流量分配的文章，文中的HWM和DUAL算法都比较实用
* [An Efficient Algorithm for Allocation of Guaranteed Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Allocation/An%20Efficient%20Algorithm%20for%20Allocation%20of%20Guaranteed%20Display%20Advertising.pdf) <br />
同样是雅虎的流量分配文章，跟上一篇文章同时发布，介绍SHALE流量分配算法

### Bidding Strategy
计算广告中广告定价，RTB过程中广告出价策略的相关问题
* [Combining Powers of Two Predictors in Optimizing Real-Time Bidding Strategy under Constrained Budget.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Combining%20Powers%20of%20Two%20Predictors%20in%20Optimizing%20Real-Time%20Bidding%20Strategy%20under%20Constrained%20Budget.pdf) <br />
国立台湾大学的文章，介绍一种基于流量选择的计算广告竞价方法，有别于传统的CTR CPC的方法，我在实践中尝试过该方法，非常有效
* [Real-Time Bidding Algorithms for Performance-Based Display Ad Allocation.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Real-Time%20Bidding%20Algorithms%20for%20Performance-Based%20Display%20Ad%20Allocation.pdf) <br />
微软的一篇基于PID反馈控制的与效果相关的竞价算法
* [Real-Time Bidding by Reinforcement Learning in Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Real-Time%20Bidding%20by%20Reinforcement%20Learning%20in%20Display%20Advertising.pdf) <br />
* [Research Frontier of Real-Time Bidding based Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Research%20Frontier%20of%20Real-Time%20Bidding%20based%20Display%20Advertising.pdf) <br />
张伟楠博士的一篇介绍竞价算法的ppt，可以非常清晰的了解该问题的主要方法

### Budget Control
广告系统中Pacing，预算控制，以及怎么把预算控制与其他模块相结合的问题
* [Budget Pacing for Targeted Online Advertisements at LinkedIn.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Budget%20Pacing%20for%20Targeted%20Online%20Advertisements%20at%20LinkedIn.pdf) <br />
linkedin的一篇非常有工程价值的解决pacing问题的文章，强烈建议计算广告系统采用此方法。
* [PID控制原理与控制算法.doc](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/PID%E6%8E%A7%E5%88%B6%E5%8E%9F%E7%90%86%E4%B8%8E%E6%8E%A7%E5%88%B6%E7%AE%97%E6%B3%95.doc) <br />
对于采用PID控制解决pacing问题，该文章是PID控制原理比较清晰的介绍文章。
* [PID控制经典培训教程.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/PID%E6%8E%A7%E5%88%B6%E7%BB%8F%E5%85%B8%E5%9F%B9%E8%AE%AD%E6%95%99%E7%A8%8B.pdf) <br />
PID控制的经典教程
* [Predicting Traffic of Online Advertising in Real-time Bidding Systems from Perspective of Demand-Side Platforms.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Predicting%20Traffic%20of%20Online%20Advertising%20in%20Real-time%20Bidding%20Systems%20from%20Perspective%20of%20Demand-Side%20Platforms.pdf) <br />
* [Real Time Bid Optimization with Smooth Budget Delivery in Online Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Real%20Time%20Bid%20Optimization%20with%20Smooth%20Budget%20Delivery%20in%20Online%20Advertising.pdf) <br />
如何将Pcaing与效果优化结合在一起，这篇文章讲的很清楚
* [Smart Pacing for Effective Online Ad Campaign Optimization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Smart%20Pacing%20for%20Effective%20Online%20Ad%20Campaign%20Optimization.pdf) <br />
跟上篇文章一样，都是雅虎同一组人写的，解决预算控制与效果结合的问题，可以跟上篇文章一起看了

### Computational Advertising Architect
广告系统的架构问题
* [Display Advertising with Real-Time Bidding (RTB) and Behavioural Targeting.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Display%20Advertising%20with%20Real-Time%20Bidding%20%28RTB%29%20and%20Behavioural%20Targeting.pdf) <br />
张伟楠博士的RTB过程所有相关算法的书，全而精，非常棒
* [Parameter Server for Distributed Machine Learning.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Parameter%20Server%20for%20Distributed%20Machine%20Learning.pdf) <br />
* [Scaling Distributed Machine Learning with the Parameter Server.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Scaling%20Distributed%20Machine%20Learning%20with%20the%20Parameter%20Server.pdf) <br />
* [大数据下的广告排序技术及实践.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%E5%A4%A7%E6%95%B0%E6%8D%AE%E4%B8%8B%E7%9A%84%E5%B9%BF%E5%91%8A%E6%8E%92%E5%BA%8F%E6%8A%80%E6%9C%AF%E5%8F%8A%E5%AE%9E%E8%B7%B5.pdf) <br />
阿里妈妈的一篇广告排序问题的ppt，模型、训练、评估都有涉及，很有工程价值
* [美团机器学习 吃喝玩乐中的算法问题.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%E7%BE%8E%E5%9B%A2%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%20%E5%90%83%E5%96%9D%E7%8E%A9%E4%B9%90%E4%B8%AD%E7%9A%84%E7%AE%97%E6%B3%95%E9%97%AE%E9%A2%98.pdf) <br />
美团王栋博士的一篇关于美团机器学习相关问题的介绍，介绍的比较全但比较粗浅，可以借此了解美团的一些机器学习问题

### CTR Prediction
CTR预估模型相关问题
* [Ad Click Prediction a View from the Trenches.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Ad%20Click%20Prediction%20a%20View%20from%20the%20Trenches.pdf) <br />
Google大名鼎鼎的用FTRL解决CTR在线预估的工程文章，非常经典。
* [Adaptive Targeting for Online Advertisement.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Adaptive%20Targeting%20for%20Online%20Advertisement.pdf) <br />
一篇比较简单但是全面的CTR预估的文章，有一定实用性
* [Deep Crossing- Web-Scale Modeling without Manually Crafted Combinatorial Features.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Deep%20Crossing-%20Web-Scale%20Modeling%20without%20Manually%20Crafted%20Combinatorial%20Features.pdf) <br />
* [Deep Interest Network for Click-Through Rate Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Deep%20Interest%20Network%20for%20Click-Through%20Rate%20Prediction.pdf) <br />
* [Deep Neural Networks for YouTube Recommendations.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Deep%20Neural%20Networks%20for%20YouTube%20Recommendations.pdf) <br />
* [Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Learning%20Piece-wise%20Linear%20Models%20from%20Large%20Scale%20Data%20for%20Ad%20Click%20Prediction.pdf) <br />
* [Logistic Regression in Rare Events Data.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Logistic%20Regression%20in%20Rare%20Events%20Data.pdf) <br />
样本稀少情况下的LR模型训练，讲的比较细
* [Practical Lessons from Predicting Clicks on Ads at Facebook.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Practical%20Lessons%20from%20Predicting%20Clicks%20on%20Ads%20at%20Facebook.pdf) <br />
Facebook的一篇非常出名的文章，GBDT＋LR/FM解决CTR预估问题，工程性很强
* [Wide & Deep Learning for Recommender Systems.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Wide%20%26%20Deep%20Learning%20for%20Recommender%20Systems.pdf) <br />

### Explore and Exploit
探索和利用问题，计算广告中非常经典的问题， 也是容易被大家忽视的问题，其实所有的广告系统都面临如何解决新广告主冷启动的问题，以及在效果不好的情况下如何探索新的优质流量的问题，希望该目录下的几篇文章能搞帮助到你。
* [A Contextual-Bandit Approach to Personalized News Article Recommendation(LinUCB).pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/A%20Contextual-Bandit%20Approach%20to%20Personalized%20News%20Article%20Recommendation%28LinUCB%29.pdf) <br />
* [A Fast and Simple Algorithm for Contextual Bandits.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/A%20Fast%20and%20Simple%20Algorithm%20for%20Contextual%20Bandits.pdf) <br />
* [An Empirical Evaluation of Thompson Sampling.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/An%20Empirical%20Evaluation%20of%20Thompson%20Sampling.pdf) <br />
* [Analysis of Thompson Sampling for the Multi-armed Bandit Problem.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Analysis%20of%20Thompson%20Sampling%20for%20the%20Multi-armed%20Bandit%20Problem.pdf) <br />
* [Bandit Algorithms Continued- UCB1.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Bandit%20Algorithms%20Continued-%20UCB1.pdf) <br />
* [Bandit based Monte-Carlo Planning.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Bandit%20based%20Monte-Carlo%20Planning.pdf) <br />
* [Customer Acquisition via Display Advertising Using MultiArmed Bandit Experiments.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Customer%20Acquisition%20via%20Display%20Advertising%20Using%20MultiArmed%20Bandit%20Experiments.pdf) <br />
* [Dynamic Online Pricing with Incomplete Information Using Multi-Armed Bandit Experiments.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Dynamic%20Online%20Pricing%20with%20Incomplete%20Information%20Using%20Multi-Armed%20Bandit%20Experiments.pdf) <br />
* [EandE.pptx](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/EandE.pptx) <br />
* [Exploitation and Exploration in a Performance based Contextual Advertising System.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Exploitation%20and%20Exploration%20in%20a%20Performance%20based%20Contextual%20Advertising%20System.pdf) <br />
* [Exploration exploitation in Go UCT for Monte-Carlo Go.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Exploration%20exploitation%20in%20Go%20UCT%20for%20Monte-Carlo%20Go.pdf) <br />
* [Exploring compact reinforcement-learning representations with linear regression.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Exploring%20compact%20reinforcement-learning%20representations%20with%20linear%20regression.pdf) <br />
* [Finite-time Analysis of the Multiarmed Bandit Problem.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Finite-time%20Analysis%20of%20the%20Multiarmed%20Bandit%20Problem.pdf) <br />
* [Hierarchical Deep Reinforcement Learning- Integrating Temporal Abstraction and Intrinsic Motivation.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Hierarchical%20Deep%20Reinforcement%20Learning-%20Integrating%20Temporal%20Abstraction%20and%20Intrinsic%20Motivation.pdf) <br />
* [INCENTIVIZING EXPLORATION IN REINFORCEMENT LEARNING WITH DEEP PREDICTIVE MODELS.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/INCENTIVIZING%20EXPLORATION%20IN%20REINFORCEMENT%20LEARNING%20WITH%20DEEP%20PREDICTIVE%20MODELS.pdf) <br />
* [Mastering the game of Go with deep neural networks and tree search.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Mastering%20the%20game%20of%20Go%20with%20deep%20neural%20networks%20and%20tree%20search.pdf) <br />
* [Multi-Armed Bandits Gittins Index and Its Calculation.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Multi-Armed%20Bandits%20Gittins%20Index%20and%20Its%20Calculation.pdf) <br />
* [On the Prior Sensitivity of Thompson Sampling.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/On%20the%20Prior%20Sensitivity%20of%20Thompson%20Sampling.pdf) <br />
* [Provable Optimal Algorithms for Generalized Linear Contextual Bandits.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Provable%20Optimal%20Algorithms%20for%20Generalized%20Linear%20Contextual%20Bandits.pdf) <br />
* [Random Forest for the Contextual Bandit Problem.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Random%20Forest%20for%20the%20Contextual%20Bandit%20Problem.pdf) <br />
* [Thompson Sampling PPT.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Thompson%20Sampling%20PPT.pdf) <br />
* [UCT算法.doc](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/UCT%E7%AE%97%E6%B3%95.doc) <br />
* [Unifying Count-Based Exploration and Intrinsic Motivation.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Unifying%20Count-Based%20Exploration%20and%20Intrinsic%20Motivation.pdf) <br />
* [Using Confidence Bounds for Exploitation-Exploration Trade-offs.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Using%20Confidence%20Bounds%20for%20Exploitation-Exploration%20Trade-offs.pdf) <br />
* [Variational Information Maximizing Exploration.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/Variational%20Information%20Maximizing%20Exploration.pdf) <br />
* [基于UCT的围棋引擎的研究与实现.doc](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/%E5%9F%BA%E4%BA%8EUCT%E7%9A%84%E5%9B%B4%E6%A3%8B%E5%BC%95%E6%93%8E%E7%9A%84%E7%A0%94%E7%A9%B6%E4%B8%8E%E5%AE%9E%E7%8E%B0.doc) <br />
* [对抗搜索、多臂老虎机问题、UCB算法.ppt](https://github.com/wzhe06/Ad-papers/blob/master/Explore%20and%20Exploit/%E5%AF%B9%E6%8A%97%E6%90%9C%E7%B4%A2%E3%80%81%E5%A4%9A%E8%87%82%E8%80%81%E8%99%8E%E6%9C%BA%E9%97%AE%E9%A2%98%E3%80%81UCB%E7%AE%97%E6%B3%95.ppt) <br />


### Factorization Machines
FM因子分解机模型的相关paper，在计算广告领域非常实用的模型
* [Factorization Machines Rendle2010.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/Factorization%20Machines%20Rendle2010.pdf) <br />
* [Fast Context-aware Recommendations with Factorization Machines.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/Fast%20Context-aware%20Recommendations%20with%20Factorization%20Machines.pdf) <br />
* [fastFM- A Library for Factorization Machines.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/fastFM-%20A%20Library%20for%20Factorization%20Machines.pdf) <br />
* [FM PPT by CMU.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/FM%20PPT%20by%20CMU.pdf) <br />
* [libfm-1.42.manual.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/libfm-1.42.manual.pdf) <br />
* [Scaling Factorization Machines to Relational Data.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/Scaling%20Factorization%20Machines%20to%20Relational%20Data.pdf) <br />


### Google Three Papers
Google三大篇，HDFS，MapReduce，BigTable，奠定大数据基础架构的三篇文章，应该读一读
* [Bigtable A Distributed Storage System for Structured Data.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Google%20Three%20Papers/Bigtable%20A%20Distributed%20Storage%20System%20for%20Structured%20Data.pdf) <br />
* [MapReduce Simplified Data Processing on Large Clusters.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Google%20Three%20Papers/MapReduce%20Simplified%20Data%20Processing%20on%20Large%20Clusters.pdf) <br />
* [The Google File System.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Google%20Three%20Papers/The%20Google%20File%20System.pdf) <br />


### Guaranteed Contracts Ads
* [A Dynamic Pricing Model for Unifying Programmatic Guarantee and Real-Time Bidding in Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/A%20Dynamic%20Pricing%20Model%20for%20Unifying%20Programmatic%20Guarantee%20and%20Real-Time%20Bidding%20in%20Display%20Advertising.pdf) <br />
* [Pricing Guaranteed Contracts in Online Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/Pricing%20Guaranteed%20Contracts%20in%20Online%20Display%20Advertising.pdf) <br />
* [Pricing Guidance in Ad Sale Negotiations The PrintAds Example.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/Pricing%20Guidance%20in%20Ad%20Sale%20Negotiations%20The%20PrintAds%20Example.pdf) <br />
* [Risk-Aware Dynamic Reserve Prices of Programmatic Guarantee in Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/Risk-Aware%20Dynamic%20Reserve%20Prices%20of%20Programmatic%20Guarantee%20in%20Display%20Advertising.pdf) <br />
* [Risk-Aware Revenue Maximization in Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/Risk-Aware%20Revenue%20Maximization%20in%20Display%20Advertising.pdf) <br />


### Machine Learning Tutorial
机器学习方面一些非常实用的学习资料
* [Deep Learning Tutorial.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/Deep%20Learning%20Tutorial.pdf) <br />
* [Rules of Machine Learning- Best Practices for ML Engineering.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/Rules%20of%20Machine%20Learning-%20Best%20Practices%20for%20ML%20Engineering.pdf) <br />
* [关联规则基本算法及其应用.doc](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E5%85%B3%E8%81%94%E8%A7%84%E5%88%99%E5%9F%BA%E6%9C%AC%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E5%BA%94%E7%94%A8.doc) <br />
* [各种回归的概念学习.doc](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E5%90%84%E7%A7%8D%E5%9B%9E%E5%BD%92%E7%9A%84%E6%A6%82%E5%BF%B5%E5%AD%A6%E4%B9%A0.doc) <br />
* [广义线性模型.ppt](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E5%B9%BF%E4%B9%89%E7%BA%BF%E6%80%A7%E6%A8%A1%E5%9E%8B.ppt) <br />
* [机器学习总图.jpg](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%80%BB%E5%9B%BE.jpg) <br />
* [贝叶斯统计学(PPT).pdf](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E8%B4%9D%E5%8F%B6%E6%96%AF%E7%BB%9F%E8%AE%A1%E5%AD%A6%28PPT%29.pdf) <br />


### Optimization Method
Online Optimization，Parallel SGD，FTRL等优化方法，很实用的一些文章
* [A Survey on Algorithms of the Regularized Convex Optimization Problem.pptx](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/A%20Survey%20on%20Algorithms%20of%20the%20Regularized%20Convex%20Optimization%20Problem.pptx) <br />
* [Follow-the-Regularized-Leader and Mirror Descent- Equivalence Theorems and L1 Regularization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Follow-the-Regularized-Leader%20and%20Mirror%20Descent-%20Equivalence%20Theorems%20and%20L1%20Regularization.pdf) <br />
* [Hogwild A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Hogwild%20A%20Lock-Free%20Approach%20to%20Parallelizing%20Stochastic%20Gradient%20Descent.pdf) <br />
* [Parallelized Stochastic Gradient Descent.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Parallelized%20Stochastic%20Gradient%20Descent.pdf) <br />
* [在线最优化求解(Online Optimization)-冯扬.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/%E5%9C%A8%E7%BA%BF%E6%9C%80%E4%BC%98%E5%8C%96%E6%B1%82%E8%A7%A3%28Online%20Optimization%29-%E5%86%AF%E6%89%AC.pdf) <br />
* [非线性规划.doc](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/%E9%9D%9E%E7%BA%BF%E6%80%A7%E8%A7%84%E5%88%92.doc) <br />


### Recommendation
推荐系统相关文章，研究不多，欢迎补充
* [基于BPR-MF算法的推荐系统设计.docx](https://github.com/wzhe06/Ad-papers/blob/master/Recommendation/%E5%9F%BA%E4%BA%8EBPR-MF%E7%AE%97%E6%B3%95%E7%9A%84%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1.docx) <br />
* [微博推荐策略平台Eros.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Recommendation/%E5%BE%AE%E5%8D%9A%E6%8E%A8%E8%8D%90%E7%AD%96%E7%95%A5%E5%B9%B3%E5%8F%B0Eros.pdf) <br />


### Topic Model
话题模型相关文章，PLSA，LDA，进行广告Context特征提取，创意优化肯定会用到Topic Model
* [Dirichlet Distribution, Dirichlet Process and Dirichlet Process Mixture(PPT).pdf](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/Dirichlet%20Distribution%2C%20Dirichlet%20Process%20and%20Dirichlet%20Process%20Mixture%28PPT%29.pdf) <br />
* [LDA数学八卦.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/LDA%E6%95%B0%E5%AD%A6%E5%85%AB%E5%8D%A6.pdf) <br />
* [Parameter estimation for text analysis.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/Parameter%20estimation%20for%20text%20analysis.pdf) <br />
* [概率语言模型及其变形系列.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/%E6%A6%82%E7%8E%87%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E5%8F%98%E5%BD%A2%E7%B3%BB%E5%88%97.pdf) <br />
* [理解共轭先验.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/%E7%90%86%E8%A7%A3%E5%85%B1%E8%BD%AD%E5%85%88%E9%AA%8C.pdf) <br />


### Transfer Learning
迁移学习相关文章，计算广告中经常遇到新广告冷启动的问题，利用迁移学习能较好解决该问题
* [A Survey on Transfer Learning.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Transfer%20Learning/A%20Survey%20on%20Transfer%20Learning.pdf) <br />
* [Scalable Hands-Free Transfer Learning for Online Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Transfer%20Learning/Scalable%20Hands-Free%20Transfer%20Learning%20for%20Online%20Advertising.pdf) <br />


### Tree Model
树模型和基于树模型的boosting模型，树模型的效果在大部分问题上非常好，在CTR，CVR模型以及特征工程方面的应用非常广，值得深入研究
* [Classification and Regression Trees.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Classification%20and%20Regression%20Trees.pdf) <br />
* [Classification and Regression Trees.ppt](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Classification%20and%20Regression%20Trees.ppt) <br />
* [Greedy Function Approximation A Gradient Boosting Machine.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Greedy%20Function%20Approximation%20A%20Gradient%20Boosting%20Machine.pdf) <br />
* [Introduction to Boosted Trees.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Introduction%20to%20Boosted%20Trees.pdf) <br />
