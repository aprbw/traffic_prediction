# Traffic Prediction

Traffic prediction is the task of predicting future traffic measurements (e.g. volume, speed, etc.) in a road network (graph), using historical data (timeseries).
Similar task, like NYC taxi and bike, are not included, because they tend to be represented as a grid, not a graph.

The papers are haphazardly selected.

## Summary

A tabular summary of paper and dataset.
The paper is reverse chronologically sorted.
The dataset is first sorted by if it is publically available
(A = publically **A**vailable;
N = **N**ot publically available),
and then number of usage.




|    model   |   venue   | published date |    A    |     A    |      A     |      A     |    A    |    A    |   A  |    A    |    A    |   N   |   N   |  N  |  N  |   N   |    N   |    N    |    N   |       |
|:----------:|:---------:|:--------------:|:-------:|:--------:|:----------:|:----------:|:-------:|:-------:|:----:|:-------:|:-------:|:-----:|:-----:|:---:|:---:|:-----:|:------:|:-------:|:------:|-------|
|            |           |                | METR-LA | PeMS-BAY | PeMS-D7(M) | PeMS-D7(L) | PeMS-04 | PeMS-08 | LOOP | PeMS-03 | PeMS-07 | INRIX | BJER4 | BJF | BRF | BRF-L | W3-715 | E5-2907 | Xiamen | TOTAL |
|   H-STGCN  |    KDD    |    23 Aug 2020 |         |          |            |            |         |         |      |         |         |       |       |     |     |       |    1   |    1    |        |   2   |
|    AGCRN   |   arXiv   |   6 Jul 2020   |         |          |            |            |    1    |    1    |      |         |         |       |       |     |     |       |        |         |        |   2   |
|   TSE-SC   | Trans-GIS |     1 Jun 2020 |    1    |     1    |            |            |         |         |      |         |         |       |       |     |     |       |        |         |        |   2   |
|    GMAN    |    AAAI   |   7 Feb 2020   |         |     1    |            |            |         |         |      |         |         |       |       |     |     |       |        |         |    1   |   2   |
|  MRA-BGCN  |    AAAI   |   7 Feb 2020   |    1    |     1    |            |            |         |         |      |         |         |       |       |     |     |       |        |         |        |   2   |
|   STSGCN   |    AAAI   |   7 Feb 2020   |         |          |            |            |    1    |    1    |      |    1    |    1    |       |       |     |     |       |        |         |        |   4   |
|    SLCNN   |    AAAI   |   7 Feb 2020   |    1    |     1    |      1     |            |         |         |      |         |         |       |       |  1  |  1  |   1   |        |         |        |   6   |
|    GWNV2   |   arXiv   |   11 Dec 2019  |    1    |     1    |            |            |         |         |      |         |         |       |       |     |     |       |        |         |        |   2   |
|   DeepGLO  |  NeurIPS  |   8 Dec 2019   |         |          |      1     |            |         |         |      |         |         |       |       |     |     |       |        |         |        |   1   |
|  TGC-LSTM  |   T-ITS   |   28 Nov 2019  |         |          |            |            |         |         |   1  |         |         |   1   |       |     |     |       |        |         |        |   2   |
|     GWN    |   IJCAI   |   10 Aug 2019  |    1    |     1    |            |            |         |         |      |         |         |       |       |     |     |       |        |         |        |   2   |
| ST-MetaNet |    KDD    |    25 Jul 19   |    1    |          |            |            |         |         |      |         |         |       |       |     |     |       |        |         |        |   1   |
|   ST-UNet  |   arXiv   |    13 Mar 19   |    1    |          |      1     |      1     |         |         |      |         |         |       |       |     |     |       |        |         |        |   3   |
|   3D-TGCN  |   arXiv   |   3 Mar 2019   |         |          |      1     |      1     |         |         |      |         |         |       |       |     |     |       |        |         |        |   2   |
|   ASTGCN   |    AAAI   |   27 Jan 2019  |         |          |            |            |    1    |    1    |      |         |         |       |       |     |     |       |        |         |        |   2   |
|    STGCN   |   IJCAI   |    13 Jul 18   |         |          |      1     |      1     |         |         |      |         |         |       |   1   |     |     |       |        |         |        |   3   |
|    DCRNN   |    ICLR   |    30 Apr 18   |    1    |     1    |            |            |         |         |      |         |         |       |       |     |     |       |        |         |        |   2   |
|  SBU-LSTM  |  UrbComp  |   14 Aug 2017  |         |          |            |            |         |         |   1  |         |         |   1   |       |     |     |       |        |         |        |   2   |
|            |           |      TOTAL     |    8    |     7    |      5     |      3     |    3    |    3    |   2  |    1    |    1    |   2   |   1   |  1  |  1  |   1   |    1   |    1    |    1   |       |





## Performance


![METR-LA MAE@60 mins](https://github.com/aprbw/traffic_prediction/blob/master/METRLA_MAE_60.PNG)

![PeMS-BAY MAE@60 mins](https://github.com/aprbw/traffic_prediction/blob/master/PeMS_BAY_60.PNG)


NOTES: The experimental setttings may vary. But the common setting is:

* Observation window = 12 timesteps

* Prediction horizon = 1 timesteps

* Prediction window = 12 timesteps

However, there are many caveats:

* Some use different batch size when testing previous models, as they increase the observation and prediction windows from previous studies, and have difficulties fitting it on GPU using the same batch size.

* Regarding adjacency matrix, some derive it using RBF from the coordinates, some use the actual connectivity, some simply learn it, and some use combinations.

* Some might also add more context, such as time of day, or day of the week.

* DeepGLO in particular, since it is treating it as a multi-channel timeseries without the spatial information, use rolling validation, 







## Dataset

[Publically available datasets and where to find them.](https://en.wikipedia.org/wiki/Fantastic_Beasts_and_Where_to_Find_Them)

* **METR-LA**
[DCRNN Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX); 
[DCRNN Baidu](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g#list/path=%2F); 
[Sensor coordinates and adjacency matrix, also from DCRNN](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph)

* California department of transportation (**Caltrans**) Performance Measurement System (**PeMS**). The website is: http://pems.dot.ca.gov/. From the website: The traffic data displayed on the map is collected in real-time from over 39,000 individual detectors. These sensors span the freeway system across all major metropolitan areas of the State of California

    * **PeMS-BAY**
    [DCRNN Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX); 
    [DCRNN Baidu](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g#list/path=%2F)
    
    [Sensor coordinates and adjacency matrix, DCRNN github](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph)

    * **PeMS-D7(M)** 
    [PKUAI26 STGCN Github](https://github.com/PKUAI26/STGCN-IJCAI-18/blob/master/data_loader/PeMS-M.zip)

    * **PeMS-D7(L)** 

    * **PeMS-04** 
    [ATSGCN Github](https://github.com/Davidham3/ASTGCN/tree/master/data);
    [Baidu with code: "p72z"](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw) 
    [From Davidham3 Github STSGCN](https://github.com/Davidham3/STSGCN)

    * **PeMS-08** 
    [ATSGCN github](https://github.com/Davidham3/ASTGCN/tree/master/data);
    [Baidu with code: "p72z"](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw) 
    [From Davidham3 github STSGCN](https://github.com/Davidham3/STSGCN)

    * **PeMS-03** 
    [Baidu with code: "p72z"](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw) 
    [From Davidham3 github STSGCN](https://github.com/Davidham3/STSGCN)

    * **PeMS-07** 
    [Baidu with code: "p72z"](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw) 
    [From Davidham3 github STSGCN](https://github.com/Davidham3/STSGCN)

    * **PeMS-SF** [UCI](https://archive.ics.uci.edu/ml/datasets/PEMS-SF)

* **LOOP** https://github.com/zhiyongc/Seattle-Loop-Data

* **MnDOT** https://www.d.umn.edu/~tkwon/TMCdata/TMCarchive.html

* **OpenITS** http://www.openits.cn/openData/index.jhtml

* **FHWA** https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/

The following datasets are not publically available:

* INRIX https://pdfs.semanticscholar.org/4b9c/9389719caff7409d9f9cee8628aef4e38b3b.pdf

* Beijing

    * BJER4
    
    * BJF
    
    * BRF
    
    * BRF-L
    
    * W3-715
    
    * E5-2907

* Xiamen https://ieeexplore.ieee.org/document/8029849

Also relevant:

* [Davidham3 list](https://github.com/Davidham3/open-traffic-datasets)




## Paper

The papers are sorted alphabetically based on model name. The citations are based on Google scholar citation on 2020 09 07.

You can find the bibtex in traffic_prediction.bib (not complete yet)




| model      | citations | venue     | published date | paper                                                                                                                                                                                                 | codes                                                     |
|------------|-----------|-----------|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| 3D-TGCN    | 12        | arXiv     | 3 Mar 2019     | [3D Graph Convolutional Networks with Temporal Graphs: A Spatial Information Free Framework For Traffic Forecasting](https://arxiv.org/abs/1903.00919)                                                |                                                           |
| AGCRN      | 3         | arXiv     | 6 Jul 2020     | [Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting](https://arxiv.org/abs/2007.02842)                                                                                            | [PyTorch](https://github.com/LeiBAI/AGCRN)                |
| ASTGCN     | 63        | AAAI      | 27 Jan 2019    | [Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting](https://www.aaai.org/ojs/index.php/AAAI/article/view/3881/3759)                                          | [PyTorch](https://github.com/guoshnBJTU/ASTGCN-r-pytorch) |
| DCRNN      | 389       | ICLR      | 30 Apr 18      | [DIFFUSION CONVOLUTIONAL RECURRENT NEURAL NETWORK: DATA-DRIVEN TRAFFIC FORECASTING](https://arxiv.org/abs/1707.01926v3)                                                                               | [TensorFlow](https://github.com/liyaguang/DCRNN)          |
| DeepGLO    | 22        | NeurIPS   | 8 Dec 19       | [Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting](https://arxiv.org/abs/1905.03806)                                                           |                                                           |
| GMAN       | 20        | AAAI      | 7 Feb 2020     | [GMAN: A Graph Multi-Attention Network for Traffic Prediction](https://arxiv.org/abs/1911.08415)                                                                                                      |                                                           |
| GWN        | 46        | IJCAI     | 10 Aug 2019    | [Graph WaveNet for Deep Spatial-Temporal Graph Modeling](https://www.ijcai.org/Proceedings/2019/0264.pdf)                                                                                             | [PyTorch](https://github.com/nnzhan/Graph-WaveNet)        |
| GWNV2      | 0         | arXiv     | 11 Dec 2019    | [Incrementally Improving Graph WaveNet Performance on Traffic Prediction](https://arxiv.org/abs/1912.07390)                                                                                           | [PyTorch](https://github.com/sshleifer/Graph-WaveNet)     |
| H-STGCN    | 0         | KDD       | 23 Aug 2020    | [Hybrid Spatio-Temporal Graph Convolutional Network: Improving Traffic Prediction with Navigation Data](https://dl.acm.org/doi/pdf/10.1145/3394486.3403358)                                           |                                                           |
| MRA-BGCN   | 7         | AAAI      | 7 Feb 2020     | [Multi-Range Attentive Bicomponent Graph Convolutional Network for Traffic Forecasting](https://aaai.org/ojs/index.php/AAAI/article/view/5758/5614)                                                   |                                                           |
| SBU-LSTM   | 157       | UrbComp   | 14 Aug 2017    | [Deep Bidirectional and Unidirectional LSTM Recurrent Neural Network for Network-wide Traffic Speed Prediction](https://arxiv.org/abs/1801.02143)                                                     |                                                           |
| ST-MetaNet | 32        | KDD       | 25 Jul 2019    | [Urban traffic prediction from spatio-temporal data using deep meta learning](https://dl.acm.org/doi/pdf/10.1145/3292500.3330884)                                                                     |                                                           |
| SLCNN      | 1         | AAAI      | 7 Feb 2020     | [Spatio-Temporal Graph Structure Learning for Traffic Forecasting](https://aaai.org/ojs/index.php/AAAI/article/view/5470/5326)                                                                        |                                                           |
| ST-UNet    | 11        | arXiv     | 13 Mar 19      | [ST-UNet: A Spatio-Temporal U-Network for Graph-structured Time Series Modeling](https://arxiv.org/abs/1903.05631)                                                                                    |                                                           |
| STGCN      | 322       | IJCAI     | 13 Jul 18      | [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875)                                                                   | [TensorFlow](https://github.com/PKUAI26/STGCN-IJCAI-18)   |
| STSGCN     | 5         | AAAI      | 7 Feb 2020     | [Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting](https://github.com/Davidham3/STSGCN/blob/master/paper/AAAI2020-STSGCN.pdf) | [MXNet](https://github.com/Davidham3/STSGCN)              |
| TGC-LSTM   | 95        | T-ITS     | 28 Nov 2019    | [Traffic Graph Convolutional Recurrent Neural Network: A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting](https://ieeexplore.ieee.org/document/8917706)                    |                                                           |
| TSE-SC     | 0         | Trans-GIS | 01 Jun 2020    | [Traffic transformer: Capturing the continuity and periodicity of time series for traffic forecasting](https://onlinelibrary.wiley.com/doi/pdf/10.1111/tgis.12644)                                    |                                                           |







## Other works

Other works that is not based on a static spatial graph of timeseries:

* [VLUC](https://arxiv.org/abs/1911.06982)

* [STDN](https://arxiv.org/pdf/1803.01254.pdf)

* [Deep Representation Learning for Trajectory Similarity Computation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8509283)

* [Curb-GAN](https://github.com/Curb-GAN/Curb-GAN) SIG KDD 2020

* [BusTr](https://dl.acm.org/doi/pdf/10.1145/3394486.3403376) SIG KDD 2020

* [DeepMove](https://dl.acm.org/doi/pdf/10.1145/3178876.3186058) WWW 2018

* https://github.com/Alro10/deep-learning-time-series

Other lists: 

* [Davidham3 list](https://github.com/Davidham3/spatio-temporal-paper-list)

* https://paperswithcode.com/task/traffic-prediction

* [A Survey on Modern Deep Neural Network for Traffic Prediction: Trends, Methods and Challenges](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9112608) IEEE TKDE 2020

* [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/pdf/1901.00596.pdf) IEEE Trans. Neural Netw. Learn. Syst. 2020
