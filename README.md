# 计算广告论文、学习资料、业界分享
动态更新工作中实现或者阅读过的计算广告相关论文、学习资料和业界分享，作为自己工作的总结，也希望能为计算广告相关行业的同学带来便利。
所有资料均来自于互联网，如有侵权，请联系王喆。同时欢迎对计算广告感兴趣的同学与我讨论相关问题，我的联系方式如下：
* Email: wzhe06@gmail.com
* LinkedIn: [王喆的LinkedIn](https://www.linkedin.com/in/zhe-wang-profile/)
* 知乎私信: [王喆的知乎](https://www.zhihu.com/people/wang-zhe-58)

**会不断加入一些重要的计算广告相关论文和资料，并去掉一些过时的或者跟计算广告不太相关的论文**
* `New!` [[Airbnb Embedding] Real-time Personalization using Embeddings for Search Ranking at Airbnb (Airbnb 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BAirbnb%20Embedding%5D%20Real-time%20Personalization%20using%20Embeddings%20for%20Search%20Ranking%20at%20Airbnb%20%28Airbnb%202018%29.pdf) <br />
2018 KDD best paper, Airbnb基于embeddding构建的实时搜索推荐系统
* `New!` [[DIEN] Deep Interest Evolution Network for Click-Through Rate Prediction (Alibaba 2019)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDIEN%5D%20Deep%20Interest%20Evolution%20Network%20for%20Click-Through%20Rate%20Prediction%20%28Alibaba%202019%29.pdf) <br />
阿里提出的深度兴趣网络（Deep Interest Network）最新改进DIEN

**其他相关资源**
* [张伟楠的RTB Papers列表](https://github.com/wnzhang/rtb-papers)<br />
* [基于Spark MLlib的CTR预估模型(LR, FM, RF, GBDT, NN, PNN)](https://github.com/wzhe06/SparkCTR) <br />
* [推荐系统相关论文和资源列表](https://github.com/wzhe06/Reco-papers) <br />
* [Honglei Zhang的推荐系统论文列表](https://github.com/hongleizhang/RSPapers)

## 目录

### Optimization Method
Online Optimization，Parallel SGD，FTRL等优化方法，实用并且能够给出直观解释的文章
* [Google Vizier A Service for Black-Box Optimization](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Google%20Vizier%20A%20Service%20for%20Black-Box%20Optimization.pdf) <br />
* [在线最优化求解(Online Optimization)-冯扬](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/%E5%9C%A8%E7%BA%BF%E6%9C%80%E4%BC%98%E5%8C%96%E6%B1%82%E8%A7%A3%28Online%20Optimization%29-%E5%86%AF%E6%89%AC.pdf) <br />
* [Hogwild A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Hogwild%20A%20Lock-Free%20Approach%20to%20Parallelizing%20Stochastic%20Gradient%20Descent.pdf) <br />
* [Parallelized Stochastic Gradient Descent](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Parallelized%20Stochastic%20Gradient%20Descent.pdf) <br />
* [A Survey on Algorithms of the Regularized Convex Optimization Problem](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/A%20Survey%20on%20Algorithms%20of%20the%20Regularized%20Convex%20Optimization%20Problem.pptx) <br />
* [Follow-the-Regularized-Leader and Mirror Descent- Equivalence Theorems and L1 Regularization](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Follow-the-Regularized-Leader%20and%20Mirror%20Descent-%20Equivalence%20Theorems%20and%20L1%20Regularization.pdf) <br />
* [A Review of Bayesian Optimization](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/A%20Review%20of%20Bayesian%20Optimization.pdf) <br />
* [Taking the Human Out of the Loop- A Review of Bayesian Optimization](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Taking%20the%20Human%20Out%20of%20the%20Loop-%20A%20Review%20of%20Bayesian%20Optimization.pdf) <br />
* [非线性规划](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/%E9%9D%9E%E7%BA%BF%E6%80%A7%E8%A7%84%E5%88%92.doc) <br />

### Topic Model
话题模型相关文章，PLSA，LDA，进行广告Context特征提取，创意优化经常会用到Topic Model
* [概率语言模型及其变形系列](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/%E6%A6%82%E7%8E%87%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E5%8F%98%E5%BD%A2%E7%B3%BB%E5%88%97.pdf) <br />
* [Parameter estimation for text analysis](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/Parameter%20estimation%20for%20text%20analysis.pdf) <br />
* [LDA数学八卦](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/LDA%E6%95%B0%E5%AD%A6%E5%85%AB%E5%8D%A6.pdf) <br />
* [Distributed Representations of Words and Phrases and their Compositionality](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/Distributed%20Representations%20of%20Words%20and%20Phrases%20and%20their%20Compositionality.pdf) <br />
* [Dirichlet Distribution, Dirichlet Process and Dirichlet Process Mixture(PPT)](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/Dirichlet%20Distribution%2C%20Dirichlet%20Process%20and%20Dirichlet%20Process%20Mixture%28PPT%29.pdf) <br />
* [理解共轭先验](https://github.com/wzhe06/Ad-papers/blob/master/Topic%20Model/%E7%90%86%E8%A7%A3%E5%85%B1%E8%BD%AD%E5%85%88%E9%AA%8C.pdf) <br />

### Google Three Papers
Google三大篇，HDFS，MapReduce，BigTable，奠定大数据基础架构的三篇文章，任何从事大数据行业的工程师都应该了解
* [MapReduce Simplified Data Processing on Large Clusters](https://github.com/wzhe06/Ad-papers/blob/master/Google%20Three%20Papers/MapReduce%20Simplified%20Data%20Processing%20on%20Large%20Clusters.pdf) <br />
* [The Google File System](https://github.com/wzhe06/Ad-papers/blob/master/Google%20Three%20Papers/The%20Google%20File%20System.pdf) <br />
* [Bigtable A Distributed Storage System for Structured Data](https://github.com/wzhe06/Ad-papers/blob/master/Google%20Three%20Papers/Bigtable%20A%20Distributed%20Storage%20System%20for%20Structured%20Data.pdf) <br />

### Factorization Machines
FM因子分解机模型的相关paper，在计算广告领域非常实用的模型
* [FM PPT by CMU](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/FM%20PPT%20by%20CMU.pdf) <br />
* [Factorization Machines Rendle2010](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/Factorization%20Machines%20Rendle2010.pdf) <br />
* [libfm-1.42.manual](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/libfm-1.42.manual.pdf) <br />
* [Scaling Factorization Machines to Relational Data](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/Scaling%20Factorization%20Machines%20to%20Relational%20Data.pdf) <br />
* [fastFM- A Library for Factorization Machines](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/fastFM-%20A%20Library%20for%20Factorization%20Machines.pdf) <br />

### Embedding
* [[Negative Sampling] Word2vec Explained Negative-Sampling Word-Embedding Method (2014)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BNegative%20Sampling%5D%20Word2vec%20Explained%20Negative-Sampling%20Word-Embedding%20Method%20%282014%29.pdf) <br />
* [[SDNE] Structural Deep Network Embedding (THU 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BSDNE%5D%20Structural%20Deep%20Network%20Embedding%20%28THU%202016%29.pdf) <br />
* [[Item2Vec] Item2Vec-Neural Item Embedding for Collaborative Filtering (Microsoft 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BItem2Vec%5D%20Item2Vec-Neural%20Item%20Embedding%20for%20Collaborative%20Filtering%20%28Microsoft%202016%29.pdf) <br />
* [[Word2Vec] Distributed Representations of Words and Phrases and their Compositionality (Google 2013)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BWord2Vec%5D%20Distributed%20Representations%20of%20Words%20and%20Phrases%20and%20their%20Compositionality%20%28Google%202013%29.pdf) <br />
* [[Word2Vec] Word2vec Parameter Learning Explained (UMich 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BWord2Vec%5D%20Word2vec%20Parameter%20Learning%20Explained%20%28UMich%202016%29.pdf) <br />
* [[Node2vec] Node2vec - Scalable Feature Learning for Networks (Stanford 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BNode2vec%5D%20Node2vec%20-%20Scalable%20Feature%20Learning%20for%20Networks%20%28Stanford%202016%29.pdf) <br />
* [[Graph Embedding] DeepWalk- Online Learning of Social Representations (SBU 2014)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BGraph%20Embedding%5D%20DeepWalk-%20Online%20Learning%20of%20Social%20Representations%20%28SBU%202014%29.pdf) <br />
* [[Airbnb Embedding] Real-time Personalization using Embeddings for Search Ranking at Airbnb (Airbnb 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BAirbnb%20Embedding%5D%20Real-time%20Personalization%20using%20Embeddings%20for%20Search%20Ranking%20at%20Airbnb%20%28Airbnb%202018%29.pdf) <br />
* [[Alibaba Embedding] Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba (Alibaba 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BAlibaba%20Embedding%5D%20Billion-scale%20Commodity%20Embedding%20for%20E-commerce%20Recommendation%20in%20Alibaba%20%28Alibaba%202018%29.pdf) <br />
* [[Word2Vec] Efficient Estimation of Word Representations in Vector Space (Google 2013)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BWord2Vec%5D%20Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space%20%28Google%202013%29.pdf) <br />
* [[LINE] LINE - Large-scale Information Network Embedding (MSRA 2015)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BLINE%5D%20LINE%20-%20Large-scale%20Information%20Network%20Embedding%20%28MSRA%202015%29.pdf) <br />

### Budget Control
广告系统中Pacing，预算控制，以及怎么把预算控制与其他模块相结合的问题
* [Budget Pacing for Targeted Online Advertisements at LinkedIn](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Budget%20Pacing%20for%20Targeted%20Online%20Advertisements%20at%20LinkedIn.pdf) <br />
* [广告系统中的智能预算控制策略](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/%E5%B9%BF%E5%91%8A%E7%B3%BB%E7%BB%9F%E4%B8%AD%E7%9A%84%E6%99%BA%E8%83%BD%E9%A2%84%E7%AE%97%E6%8E%A7%E5%88%B6%E7%AD%96%E7%95%A5.pdf) <br />
* [Predicting Traffic of Online Advertising in Real-time Bidding Systems from Perspective of Demand-Side Platforms](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Predicting%20Traffic%20of%20Online%20Advertising%20in%20Real-time%20Bidding%20Systems%20from%20Perspective%20of%20Demand-Side%20Platforms.pdf) <br />
* [Real Time Bid Optimization with Smooth Budget Delivery in Online Advertising](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Real%20Time%20Bid%20Optimization%20with%20Smooth%20Budget%20Delivery%20in%20Online%20Advertising.pdf) <br />
* [PID控制经典培训教程](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/PID%E6%8E%A7%E5%88%B6%E7%BB%8F%E5%85%B8%E5%9F%B9%E8%AE%AD%E6%95%99%E7%A8%8B.pdf) <br />
* [PID控制原理与控制算法](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/PID%E6%8E%A7%E5%88%B6%E5%8E%9F%E7%90%86%E4%B8%8E%E6%8E%A7%E5%88%B6%E7%AE%97%E6%B3%95.doc) <br />
* [Smart Pacing for Effective Online Ad Campaign Optimization](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Smart%20Pacing%20for%20Effective%20Online%20Ad%20Campaign%20Optimization.pdf) <br />

### Tree Model
树模型和基于树模型的boosting模型，树模型的效果在大部分问题上非常好，在CTR，CVR预估及特征工程方面的应用非常广
* [Introduction to Boosted Trees](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Introduction%20to%20Boosted%20Trees.pdf) <br />
* [Classification and Regression Trees](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Classification%20and%20Regression%20Trees.pdf) <br />
* [Greedy Function Approximation A Gradient Boosting Machine](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Greedy%20Function%20Approximation%20A%20Gradient%20Boosting%20Machine.pdf) <br />
* [Classification and Regression Trees](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Classification%20and%20Regression%20Trees.ppt) <br />

### Guaranteed Contracts Ads
事实上，现在很多大的媒体主仍是合约广告系统，合约广告系统的在线分配，Yield Optimization，以及定价问题都是非常重要且有挑战性的问题
* [A Dynamic Pricing Model for Unifying Programmatic Guarantee and Real-Time Bidding in Display Advertising](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/A%20Dynamic%20Pricing%20Model%20for%20Unifying%20Programmatic%20Guarantee%20and%20Real-Time%20Bidding%20in%20Display%20Advertising.pdf) <br />
* [Pricing Guaranteed Contracts in Online Display Advertising](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/Pricing%20Guaranteed%20Contracts%20in%20Online%20Display%20Advertising.pdf) <br />
* [Risk-Aware Dynamic Reserve Prices of Programmatic Guarantee in Display Advertising](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/Risk-Aware%20Dynamic%20Reserve%20Prices%20of%20Programmatic%20Guarantee%20in%20Display%20Advertising.pdf) <br />
* [Pricing Guidance in Ad Sale Negotiations The PrintAds Example](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/Pricing%20Guidance%20in%20Ad%20Sale%20Negotiations%20The%20PrintAds%20Example.pdf) <br />
* [Risk-Aware Revenue Maximization in Display Advertising](https://github.com/wzhe06/Ad-papers/blob/master/Guaranteed%20Contracts%20Ads/Risk-Aware%20Revenue%20Maximization%20in%20Display%20Advertising.pdf) <br />

### Classic CTR Prediction
* [[LR] Predicting Clicks - Estimating the Click-Through Rate for New Ads (Microsoft 2007)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BLR%5D%20Predicting%20Clicks%20-%20Estimating%20the%20Click-Through%20Rate%20for%20New%20Ads%20%28Microsoft%202007%29.pdf) <br />
* [[FFM] Field-aware Factorization Machines for CTR Prediction (Criteo 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BFFM%5D%20Field-aware%20Factorization%20Machines%20for%20CTR%20Prediction%20%28Criteo%202016%29.pdf) <br />
* [[GBDT+LR] Practical Lessons from Predicting Clicks on Ads at Facebook (Facebook 2014)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BGBDT%2BLR%5D%20Practical%20Lessons%20from%20Predicting%20Clicks%20on%20Ads%20at%20Facebook%20%28Facebook%202014%29.pdf) <br />
* [[PS-PLM] Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction (Alibaba 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BPS-PLM%5D%20Learning%20Piece-wise%20Linear%20Models%20from%20Large%20Scale%20Data%20for%20Ad%20Click%20Prediction%20%28Alibaba%202017%29.pdf) <br />
* [[FTRL] Ad Click Prediction a View from the Trenches (Google 2013)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BFTRL%5D%20Ad%20Click%20Prediction%20a%20View%20from%20the%20Trenches%20%28Google%202013%29.pdf) <br />
* [[FM] Fast Context-aware Recommendations with Factorization Machines (UKON 2011)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BFM%5D%20Fast%20Context-aware%20Recommendations%20with%20Factorization%20Machines%20%28UKON%202011%29.pdf) <br />

### Bidding Strategy
计算广告中广告定价，RTB过程中广告出价策略的相关问题
* [Research Frontier of Real-Time Bidding based Display Advertising](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Research%20Frontier%20of%20Real-Time%20Bidding%20based%20Display%20Advertising.pdf) <br />
* [Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Budget%20Constrained%20Bidding%20by%20Model-free%20Reinforcement%20Learning%20in%20Display%20Advertising.pdf) <br />
* [Real-Time Bidding with Multi-Agent Reinforcement Learning in Display Advertising](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Real-Time%20Bidding%20with%20Multi-Agent%20Reinforcement%20Learning%20in%20Display%20Advertising.pdf) <br />
* [Real-Time Bidding by Reinforcement Learning in Display Advertising](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Real-Time%20Bidding%20by%20Reinforcement%20Learning%20in%20Display%20Advertising.pdf) <br />
* [Combining Powers of Two Predictors in Optimizing Real-Time Bidding Strategy under Constrained Budget](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Combining%20Powers%20of%20Two%20Predictors%20in%20Optimizing%20Real-Time%20Bidding%20Strategy%20under%20Constrained%20Budget.pdf) <br />
* [Bid-aware Gradient Descent for Unbiased Learning with Censored Data in Display Advertising](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Bid-aware%20Gradient%20Descent%20for%20Unbiased%20Learning%20with%20Censored%20Data%20in%20Display%20Advertising.pdf) <br />
* [Optimized Cost per Click in Taobao Display Advertising](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Optimized%20Cost%20per%20Click%20in%20Taobao%20Display%20Advertising.pdf) <br />
* [Real-Time Bidding Algorithms for Performance-Based Display Ad Allocation](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Real-Time%20Bidding%20Algorithms%20for%20Performance-Based%20Display%20Ad%20Allocation.pdf) <br />
* [Deep Reinforcement Learning for Sponsored Search Real-time Bidding](https://github.com/wzhe06/Ad-papers/blob/master/Bidding%20Strategy/Deep%20Reinforcement%20Learning%20for%20Sponsored%20Search%20Real-time%20Bidding.pdf) <br />

### Computational Advertising Architect
广告系统的架构问题
* [[TensorFlow Whitepaper]TensorFlow- Large-Scale Machine Learning on Heterogeneous Distributed Systems](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%5BTensorFlow%20Whitepaper%5DTensorFlow-%20Large-Scale%20Machine%20Learning%20on%20Heterogeneous%20Distributed%20Systems.pdf) <br />
* [大数据下的广告排序技术及实践](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%E5%A4%A7%E6%95%B0%E6%8D%AE%E4%B8%8B%E7%9A%84%E5%B9%BF%E5%91%8A%E6%8E%92%E5%BA%8F%E6%8A%80%E6%9C%AF%E5%8F%8A%E5%AE%9E%E8%B7%B5.pdf) <br />
* [美团机器学习 吃喝玩乐中的算法问题](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%E7%BE%8E%E5%9B%A2%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%20%E5%90%83%E5%96%9D%E7%8E%A9%E4%B9%90%E4%B8%AD%E7%9A%84%E7%AE%97%E6%B3%95%E9%97%AE%E9%A2%98.pdf) <br />
* [[Parameter Server]Scaling Distributed Machine Learning with the Parameter Server](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%5BParameter%20Server%5DScaling%20Distributed%20Machine%20Learning%20with%20the%20Parameter%20Server.pdf) <br />
* [Display Advertising with Real-Time Bidding (RTB) and Behavioural Targeting](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Display%20Advertising%20with%20Real-Time%20Bidding%20%28RTB%29%20and%20Behavioural%20Targeting.pdf) <br />
* [A Comparison of Distributed Machine Learning Platforms](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/A%20Comparison%20of%20Distributed%20Machine%20Learning%20Platforms.pdf) <br />
* [Efficient Query Evaluation using a Two-Level Retrieval Process](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Efficient%20Query%20Evaluation%20using%20a%20Two-Level%20Retrieval%20Process.pdf) <br />
* [[TensorFlow Whitepaper]TensorFlow- A System for Large-Scale Machine Learning](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%5BTensorFlow%20Whitepaper%5DTensorFlow-%20A%20System%20for%20Large-Scale%20Machine%20Learning.pdf) <br />
* [[Parameter Server]Parameter Server for Distributed Machine Learning](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%5BParameter%20Server%5DParameter%20Server%20for%20Distributed%20Machine%20Learning.pdf) <br />
* [Overlapping Experiment Infrastructure More, Better, Faster Experimentation](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Overlapping%20Experiment%20Infrastructure%20More%2C%20Better%2C%20Faster%20Experimentation.pdf) <br />

### Machine Learning Tutorial
机器学习方面一些非常实用的学习资料
* [各种回归的概念学习](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E5%90%84%E7%A7%8D%E5%9B%9E%E5%BD%92%E7%9A%84%E6%A6%82%E5%BF%B5%E5%AD%A6%E4%B9%A0.doc) <br />
* [机器学习总图](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%80%BB%E5%9B%BE.jpg) <br />
* [Efficient Estimation of Word Representations in Vector Space](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space.pdf) <br />
* [Rules of Machine Learning- Best Practices for ML Engineering](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/Rules%20of%20Machine%20Learning-%20Best%20Practices%20for%20ML%20Engineering.pdf) <br />
* [An introduction to ROC analysis](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/An%20introduction%20to%20ROC%20analysis.pdf) <br />
* [Deep Learning Tutorial](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/Deep%20Learning%20Tutorial.pdf) <br />
* [广义线性模型](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E5%B9%BF%E4%B9%89%E7%BA%BF%E6%80%A7%E6%A8%A1%E5%9E%8B.ppt) <br />
* [贝叶斯统计学(PPT)](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E8%B4%9D%E5%8F%B6%E6%96%AF%E7%BB%9F%E8%AE%A1%E5%AD%A6%28PPT%29.pdf) <br />
* [关联规则基本算法及其应用](https://github.com/wzhe06/Ad-papers/blob/master/Machine%20Learning%20Tutorial/%E5%85%B3%E8%81%94%E8%A7%84%E5%88%99%E5%9F%BA%E6%9C%AC%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E5%BA%94%E7%94%A8.doc) <br />

### Transfer Learning
迁移学习相关文章，计算广告中经常遇到新广告冷启动的问题，利用迁移学习能较好解决该问题
* [[Multi-Task]An Overview of Multi-Task Learning in Deep Neural Networks](https://github.com/wzhe06/Ad-papers/blob/master/Transfer%20Learning/%5BMulti-Task%5DAn%20Overview%20of%20Multi-Task%20Learning%20in%20Deep%20Neural%20Networks.pdf) <br />
* [Scalable Hands-Free Transfer Learning for Online Advertising](https://github.com/wzhe06/Ad-papers/blob/master/Transfer%20Learning/Scalable%20Hands-Free%20Transfer%20Learning%20for%20Online%20Advertising.pdf) <br />
* [A Survey on Transfer Learning](https://github.com/wzhe06/Ad-papers/blob/master/Transfer%20Learning/A%20Survey%20on%20Transfer%20Learning.pdf) <br />

### Deep Learning CTR Prediction
* [[DCN] Deep & Cross Network for Ad Click Predictions (Stanford 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDCN%5D%20Deep%20%26%20Cross%20Network%20for%20Ad%20Click%20Predictions%20%28Stanford%202017%29.pdf) <br />
* [[Deep Crossing] Deep Crossing - Web-Scale Modeling without Manually Crafted Combinatorial Features (Microsoft 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDeep%20Crossing%5D%20Deep%20Crossing%20-%20Web-Scale%20Modeling%20without%20Manually%20Crafted%20Combinatorial%20Features%20%28Microsoft%202016%29.pdf) <br />
* [[PNN] Product-based Neural Networks for User Response Prediction (SJTU 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BPNN%5D%20Product-based%20Neural%20Networks%20for%20User%20Response%20Prediction%20%28SJTU%202016%29.pdf) <br />
* [[DIN] Deep Interest Network for Click-Through Rate Prediction (Alibaba 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDIN%5D%20Deep%20Interest%20Network%20for%20Click-Through%20Rate%20Prediction%20%28Alibaba%202018%29.pdf) <br />
* [[ESMM] Entire Space Multi-Task Model - An Effective Approach for Estimating Post-Click Conversion Rate (Alibaba 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BESMM%5D%20Entire%20Space%20Multi-Task%20Model%20-%20An%20Effective%20Approach%20for%20Estimating%20Post-Click%20Conversion%20Rate%20%28Alibaba%202018%29.pdf) <br />
* [[Wide & Deep] Wide & Deep Learning for Recommender Systems (Google 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BWide%20%26%20Deep%5D%20Wide%20%26%20Deep%20Learning%20for%20Recommender%20Systems%20%28Google%202016%29.pdf) <br />
* [[xDeepFM] xDeepFM - Combining Explicit and Implicit Feature Interactions for Recommender Systems (USTC 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BxDeepFM%5D%20xDeepFM%20-%20Combining%20Explicit%20and%20Implicit%20Feature%20Interactions%20for%20Recommender%20Systems%20%28USTC%202018%29.pdf) <br />
* [[Image CTR] Image Matters - Visually modeling user behaviors using Advanced Model Server (Alibaba 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BImage%20CTR%5D%20Image%20Matters%20-%20Visually%20modeling%20user%20behaviors%20using%20Advanced%20Model%20Server%20%28Alibaba%202018%29.pdf) <br />
* [[AFM] Attentional Factorization Machines - Learning the Weight of Feature Interactions via Attention Networks (ZJU 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BAFM%5D%20Attentional%20Factorization%20Machines%20-%20Learning%20the%20Weight%20of%20Feature%20Interactions%20via%20Attention%20Networks%20%28ZJU%202017%29.pdf) <br />
* [[DIEN] Deep Interest Evolution Network for Click-Through Rate Prediction (Alibaba 2019)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDIEN%5D%20Deep%20Interest%20Evolution%20Network%20for%20Click-Through%20Rate%20Prediction%20%28Alibaba%202019%29.pdf) <br />
* [[DSSM] Learning Deep Structured Semantic Models for Web Search using Clickthrough Data (UIUC 2013)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDSSM%5D%20Learning%20Deep%20Structured%20Semantic%20Models%20for%20Web%20Search%20using%20Clickthrough%20Data%20%28UIUC%202013%29.pdf) <br />
* [[FNN] Deep Learning over Multi-field Categorical Data (UCL 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BFNN%5D%20Deep%20Learning%20over%20Multi-field%20Categorical%20Data%20%28UCL%202016%29.pdf) <br />
* [[DeepFM] A Factorization-Machine based Neural Network for CTR Prediction (HIT-Huawei 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDeepFM%5D%20A%20Factorization-Machine%20based%20Neural%20Network%20for%20CTR%20Prediction%20%28HIT-Huawei%202017%29.pdf) <br />
* [[NFM] Neural Factorization Machines for Sparse Predictive Analytics (NUS 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BNFM%5D%20Neural%20Factorization%20Machines%20for%20Sparse%20Predictive%20Analytics%20%28NUS%202017%29.pdf) <br />

### Exploration and Exploitation
探索和利用，计算广告中非常经典，也是容易被大家忽视的问题，其实所有的广告系统都面临如何解决新广告主冷启动，以及在效果不好的情况下如何探索新的优质流量的问题，希望该目录下的几篇文章能够帮助到你
* [An Empirical Evaluation of Thompson Sampling](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/An%20Empirical%20Evaluation%20of%20Thompson%20Sampling.pdf) <br />
* [Dynamic Online Pricing with Incomplete Information Using Multi-Armed Bandit Experiments](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Dynamic%20Online%20Pricing%20with%20Incomplete%20Information%20Using%20Multi-Armed%20Bandit%20Experiments.pdf) <br />
* [广告系统中的探索与利用算法](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/%E5%B9%BF%E5%91%8A%E7%B3%BB%E7%BB%9F%E4%B8%AD%E7%9A%84%E6%8E%A2%E7%B4%A2%E4%B8%8E%E5%88%A9%E7%94%A8%E7%AE%97%E6%B3%95.pdf) <br />
* [Finite-time Analysis of the Multiarmed Bandit Problem](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Finite-time%20Analysis%20of%20the%20Multiarmed%20Bandit%20Problem.pdf) <br />
* [A Fast and Simple Algorithm for Contextual Bandits](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/A%20Fast%20and%20Simple%20Algorithm%20for%20Contextual%20Bandits.pdf) <br />
* [Customer Acquisition via Display Advertising Using MultiArmed Bandit Experiments](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Customer%20Acquisition%20via%20Display%20Advertising%20Using%20MultiArmed%20Bandit%20Experiments.pdf) <br />
* [Mastering the game of Go with deep neural networks and tree search](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Mastering%20the%20game%20of%20Go%20with%20deep%20neural%20networks%20and%20tree%20search.pdf) <br />
* [Exploring compact reinforcement-learning representations with linear regression](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Exploring%20compact%20reinforcement-learning%20representations%20with%20linear%20regression.pdf) <br />
* [Incentivizting Exploration in Reinforcement Learning with Deep Predictive Models](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Incentivizting%20Exploration%20in%20Reinforcement%20Learning%20with%20Deep%20Predictive%20Models.pdf) <br />
* [Bandit Algorithms Continued- UCB1](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Bandit%20Algorithms%20Continued-%20UCB1.pdf) <br />
* [A Contextual-Bandit Approach to Personalized News Article Recommendation(LinUCB)](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/A%20Contextual-Bandit%20Approach%20to%20Personalized%20News%20Article%20Recommendation%28LinUCB%29.pdf) <br />
* [Exploitation and Exploration in a Performance based Contextual Advertising System](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Exploitation%20and%20Exploration%20in%20a%20Performance%20based%20Contextual%20Advertising%20System.pdf) <br />
* [Bandit based Monte-Carlo Planning](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Bandit%20based%20Monte-Carlo%20Planning.pdf) <br />
* [Random Forest for the Contextual Bandit Problem](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Random%20Forest%20for%20the%20Contextual%20Bandit%20Problem.pdf) <br />
* [Unifying Count-Based Exploration and Intrinsic Motivation](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Unifying%20Count-Based%20Exploration%20and%20Intrinsic%20Motivation.pdf) <br />
* [Analysis of Thompson Sampling for the Multi-armed Bandit Problem](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Analysis%20of%20Thompson%20Sampling%20for%20the%20Multi-armed%20Bandit%20Problem.pdf) <br />
* [Thompson Sampling PPT](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Thompson%20Sampling%20PPT.pdf) <br />
* [Hierarchical Deep Reinforcement Learning- Integrating Temporal Abstraction and Intrinsic Motivation](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Hierarchical%20Deep%20Reinforcement%20Learning-%20Integrating%20Temporal%20Abstraction%20and%20Intrinsic%20Motivation.pdf) <br />
* [Exploration and Exploitation Problem by Wang Zhe](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Exploration%20and%20Exploitation%20Problem%20by%20Wang%20Zhe.pptx) <br />
* [Exploration exploitation in Go UCT for Monte-Carlo Go](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Exploration%20exploitation%20in%20Go%20UCT%20for%20Monte-Carlo%20Go.pdf) <br />
* [对抗搜索、多臂老虎机问题、UCB算法](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/%E5%AF%B9%E6%8A%97%E6%90%9C%E7%B4%A2%E3%80%81%E5%A4%9A%E8%87%82%E8%80%81%E8%99%8E%E6%9C%BA%E9%97%AE%E9%A2%98%E3%80%81UCB%E7%AE%97%E6%B3%95.ppt) <br />
* [Using Confidence Bounds for Exploitation-Exploration Trade-offs](https://github.com/wzhe06/Ad-papers/blob/master/Exploration%20and%20Exploitation/Using%20Confidence%20Bounds%20for%20Exploitation-Exploration%20Trade-offs.pdf) <br />

### Allocation
广告流量的分配问题
* [An Efficient Algorithm for Allocation of Guaranteed Display Advertising](https://github.com/wzhe06/Ad-papers/blob/master/Allocation/An%20Efficient%20Algorithm%20for%20Allocation%20of%20Guaranteed%20Display%20Advertising.pdf) <br />
* [Ad Serving Using a Compact Allocation Plan](https://github.com/wzhe06/Ad-papers/blob/master/Allocation/Ad%20Serving%20Using%20a%20Compact%20Allocation%20Plan.pdf) <br />
