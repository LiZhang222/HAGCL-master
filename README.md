# Homophilic-Aware Graph Contrastive Learning (HAGCL) 

[![Status: Accepted](https://img.shields.io/badge/Status-Accepted-brightgreen)](https://www.sciencedirect.com/science/article/pii/S0031320326002931)
[![Journal: Pattern Recognition](https://img.shields.io/badge/PatternRecognition-2026-blue)](https://www.sciencedirect.com/science/article/pii/S0031320326002931)
[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-orange)](https://www.sciencedirect.com/science/article/pii/S0031320326002931)

Official implementation of “Homophilic-Aware Graph Contrastive Learning” (Pattern Recognition, 2026).  
Authors:  Li Zhang, Hua Mao, Wai Lok Woo, Jie Chen

- Paper: https://www.sciencedirect.com/science/article/pii/S0031320326002931

---
## Abstract
Graph contrastive learning (GCL) has become increasingly popular in unsupervised graph representation learning (UGRL). Currently, most existing GCL approaches implicitly rely on the homophily assumption. These approaches encounter two challenges when adapting to heterophily issues. First, they often suffer from losing intrinsic structural information owing to the separate encoding processes that are applied to homophilic and heterophilic views. Second, they overlook the importance of different subgraphs when disentangling a heterophilic graph into several relational graphs. In this paper, we propose a homophilic-aware GCL (HAGCL) method that learns informative node representations for both homophilic and heterophilic graphs in a unified framework. Specifically, we first introduce a homophilic-aware augmentation scheme for generating more effective views with a higher homophily ratio. This scheme potentially enables the learned node representations to be more informative and robust. We then introduce a discriminative relation and frequency adaptation (DRFA) module to address the disentangling problem. This DRFA module can adaptively capture diverse relations and frequency signals while discriminatively fusing multiple relational graphs. Extensive experimental results that were obtained on homophilic and heterophilic datasets demonstrate the effectiveness of the proposed HAGCL method.

---


## Getting Started
### test:
```python

python main_transductive.py --dataset <dataset_name>  --load_model
```

### train:
```python

python main_transductive.py --dataset <dataset_name>
```


---

## License and Usage
- This code is released for research purposes.  

---

## Contact
- For questions or issues, please open a GitHub issue in this repository.



