### 1. 度学习需要的技术栈学习demo:
1. Ray:分布式机器学习框架:Ray Core(基础的分布式能力),Ray Train(分布式训练),Ray Data(分布式数据),Ray Tune(超参数调整), Ray Serve(分布式推理)...   
link: https://zw2tnvaduvx.feishu.cn/wiki/EPBFwBYkhiXSahkvddzczNfvnyc?from=from_copylink
2. triton: 是一种可以替代navie cuda编程模式的并行编程框架。与cuda相比，triton遵循的是“单程序，多数据，且每个程序粒度则是block而不是线程"
link: https://triton-lang.cn/main/python-api/triton-semantics.html

### 2. 深度学习的特性
1. 深度学习框架中的广播: 广播允许在不同形状的张量上执行操作，通过自动将它们的形状扩展兼容的大小，而无需复制数据。这遵循以下规则:
1.1. 如果一个张量形状维度较少，则在左侧用1填充，直到两个张量具有相同的维度数: ((3,4),(5,3,4))-> ((1, 3, 4), (5, 3, 4))
1.2. 如果两个维度相等，或者其中一个为 1，则它们是兼容的。值为 1 的维度将被扩展以匹配另一个张量的维度。((1, 3, 4), (5, 3, 4)) -> ((5, 3, 4), (5, 3, 4))