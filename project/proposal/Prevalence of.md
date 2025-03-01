# Prevalence of....





Modern practice for training classification deepnets involves a *Terminal Phase of Training* (TPT), which begins at the epoch where training error first vanishes; During TPT, the training error stays effectively zero while training loss is pushed towards zero. Direct measurements of TPT, for three prototypical deepnet architectures and across seven canonical classification datasets, expose a pervasive inductive bias we call *Neural Collapse*, involving four deeply interconnected phenomena: (NC1) Cross-example within-class variability of last-layer training activations collapses to zero, as the individual activations themselves collapse to their class-means; (NC2) The class-means collapse to the vertices of a Simplex Equiangular Tight Frame (ETF); (NC3) Up to rescaling, the last-layer classifiers collapse to the class-means, or in other words to the Simplex ETF, i.e. to a *self-dual* configuration; (NC4) For a given activation, the classifier’s decision collapses to simply choosing whichever class has the closest train class-mean, i.e. the Nearest Class-Center (NCC) decision rule. The symmetric and very simple geometry induced by the TPT confers important benefits, including better generalization performance, better robustness, and better interpretability.

训练分类深度网的现代做法包括*训练的最终阶段*（TPT），它始于训练误差首次消失的那个时点；在TPT期间，训练误差实际上保持为零，而训练损失则趋向于零。针对三种原型深度网络架构和七个典型分类数据集对 TPT 进行的直接测量揭示了一种普遍存在的归纳偏差，我们称之为 “神经崩溃”（*Neural Collapse*），其中涉及四种相互关联的现象： (NC1) 最后一层训练激活的跨样本类内变异性坍缩为零，因为单个激活本身坍缩为其类均值；(NC2) 类均值坍缩为简单等相紧帧（ETF）的顶点；(NC3) 在重新缩放之前，最后一层分类器坍缩为类均值，或者换句话说，坍缩为简单等相紧帧（ETF），即 “*自渡”；(NC4) 在重新缩放之后，最后一层分类器坍缩为类均值，或者换句话说，坍缩为简单等相紧帧（ETF），即 “*自渡”。 (NC4) 对于给定的激活，分类器的决策会折叠为简单地选择最接近列车类均值的类，即最近类中心（NCC）决策规则。由 TPT 诱导的对称和非常简单的几何形状带来了重要的好处，包括更好的泛化性能、更好的鲁棒性和更好的可解释性。















# Geometric Analysis

















