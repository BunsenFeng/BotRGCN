### BotRGCN

#### Introduction to BotRGCN
Twitter users operated by automated programs, also known as bots, have increased their appearance recently and induced undesirable social effects. While extensive research efforts have been devoted to the task of Twitter bot detection, previous methods leverage only a small fraction of user semantic and profile information, which leads to their failure in identifying bots that exploit multi-modal user information to disguise as genuine users. Apart from that, the state-of-the-art bot detectors fail to leverage user follow relationships and the graph structure it forms. As a result, these methods fall short of capturing new generations of Twitter bots that act in groups and seem genuine individually. To address these two challenges of Twitter bot detection, we propose BotRGCN, which is short for Bot detection with Relational Graph Convolutional Networks. BotRGCN addresses the challenge of community by constructing a heterogeneous graph from follow relationships and apply relational graph convolutional networks to the Twittersphere. Apart from that, BotRGCN makes use of multi-modal user semantic and property information to avoid feature engineering and augment its ability to capture bots with diversified disguise. Extensive experiments demonstrate that BotRGCN outperforms competitive baselines on a comprehensive benchmark TwiBot-20 which provides follow relationships. BotRGCN is also proved to effectively leverage three modals of user information, namely semantic, property and neighborhood information, to boost bot detection performance.

#### Affiliated Paper
The affiliated paper of this repository, ['BotRGCN: Twitter Bot Detection with Relational Graph Convolutional Networks'](https://arxiv.org/abs/2106.13092), is accepted at ASONAM'21. Work in progress.

#### Data Description
More details at [TwiBot-20 data](https://github.com/GabrielHam/TwiBot-20.) , please download 'Twibot-20.zip' to the folder which also contains 'Dataset.py' and unzip it here.

#### Code Description

- Dataset.py

  - ```python
    class Twibot20(self,root='./Data/,device='cpu',process=True,save=True)
    ```

    - `root` - the folder where the processed data is saved , the default folder is './Data' , which has already been created
    - `save` - whether to save the processed data or not (set it to True can save you a lot of time if you want to run this model for further ablation study)
    - `process` - If you have already saved the processed data,set it to True

- model.py

  - BotRGCN - the standard BotRGCN
  - BotRGCN1 - using the description feature alone
  - BotRGCN2 - using the tweets feature alone
  - BotRGCN3 - using the numerical properties feature alone
  - BotRGCN4 - using the categorical properties feature alone
  - BotRGCN12 - using the description feature + the tweets feature
  - BotRGCN34 - using the numerical properties feature + the categorical properties feature
  - BotGCN - replace the RGCNConv layers with GCNConv layers
  - BotGAT - replace the RGCNConv layers with GATConv layers
  - BotRGCN_4layers - BotRGCN with 4 RGCNConv layers
  - BotRGCN_8layers - BotRGCN with 8 RGCNConv layers
