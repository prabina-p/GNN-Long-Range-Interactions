# GNN-Long-Range-Interactions

**Project Overview**

This project provides an exploration on baseline Graph Neural Network (GNN) models such as Graph Convolutional Network (GCN), Graph Isomorphism Network (GIN), and Graph Attention Network (GATv2), as well as the transformer-based model GraphGPS for graph representation learning. We delve deep into four datasets: conventional datasets like Cora, IMDB, and Enzymes; and long-range benchmark dataset like PascalVOC-SP. Cora and PascalVOC-SP is being used here for node classification task, whereas IMDB and Enzymes is being used for graph classification task.


**Structure**

```python
Project
├── README.md
├── output.txt
├── requirements.txt
├── scripts.py
└── examples.ipynb
```

**Project Usage**

1. Clone the repo to your local machine
2. Open your terminal and cd into the directory of the cloned repo
3. On the terminal, run ```pip install -r requirements.txt``` to obtain necessary packages to run the code.
4. Next, run ```python scripts.py```. This will run the four models on all four datasets. You can have the output.txt file open and view accuracy change in real time. Keep in mind that since epochs is 500, it will take roughly an hour to two to complete running this code.


**References**

- https://github.com/qbxlvnf11/graph-neural-networks-for-graph-classification
- https://github.com/graphdeeplearning/graphtransformer/tree/main
- https://github.com/rampasek/GraphGPS
- https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html
- https://github.com/tech-srl/bottleneck/