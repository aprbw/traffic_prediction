# Traffic Prediction

Traffic prediction is the task of predicting future traffic measurements (e.g. volume, speed, etc.) in a road network (graph), using historical data (timeseries).

Things are usually better defined through exclusions, so here are similar things that I do not include:

* NYC taxi and bike (and other similar datsets, like uber), are not included, because they tend to be represented as a grid, not a graph.

* Predicting human mobility, either indoors, or through checking-in in Point of Interest (POI), or through a transport network.

* Predicting trajectory.

* Predicting the movement of individual cars through sensors for the purpose of self-driving car.

* Traffic data imputations.

* Traffic anomaly detections.

The papers are haphazardly selected.

## Summary

A tabular summary of paper and publically available datasets.
The paper is reverse chronologically sorted.
NO GUARANTEE is made that this table is complete or accurate (please raise an issue if you spot any error).


| paper | venue | published date | # other datsets | METR-LA | PeMS-BAY | PeMS-D7(M) | PeMS-D7(L) | PeMS-04 | PeMS-08 | LOOP | SZ-taxi | Los-loop | PeMS-03 | PeMS-07 | PeMS-I-405 | PeMS-04(S) | TOTAL open |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|-|:-:|
|  |  | TOTAL |  | 38 | 28 | 6 | 3 | 3 | 3 | 3 | 2 | 2 | 1 | 1 | 1 | 1 | 95 |
| SCPT | ArXiv | 9 May 23 | 1 | 1 | 1 | 1 |  |  |  |  |  |  |  |  |  |  | 4 |
| G-SWaN | IoTDI | 9 May 23 |   | 1 | 1 |  |  | 1 | 1 |  |  |  |  |  |  |  | 4 |
| GTS | ICLR | 4 May 21 | 1 | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| FASTGNN | TII | 29 Jan 21 |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  | 1 |
| HetGAT | JAIHC | 23 Jan 21 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| GST-GAT | IEEE Access | 6 Jan 21 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| CLGRN | arXiv | 4 Jan 21 | 3 | 1 |  |  |  |  |  |  |  |  |  |  |  |  | 1 |
| DKFN | SIGSPATIAL | 3 Nov 20 |  | 1 |  |  |  |  |  | 1 |  |  |  |  |  |  | 2 |
| STGAM | CISP-BMEI | 17 Oct 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| ARNN | Nat. Commun | 11 Sept 20 |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  | 1 |
| ST-TrafficNet | ELECGJ | 9 Sept 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| M2 | J. AdHoc | 1 Sept 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| H-STGCN | KDD | 23 Aug 20 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| SGMN | J. TRC | 20 Aug 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| GDRNN | NTU | 16 Aug 20 |  | 1 |  |  |  |  |  |  |  |  |  |  | 1 |  | 2 |
| ISTD-GCN | arXiv | 10 Aug 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| GTS | UCONN | 3 Aug 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| FC-GAGA | arXiv | 30 Jul 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| STGAT | IEEE Access | 22 Jul 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| STNN | T-ITS | 16 Jul 20 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| AGCRN | arXiv | 6 Jul 20 |  |  |  |  |  | 1 | 1 |  |  |  |  |  |  |  | 2 |
| GWNN-LSTM | 	J. Phys. Conf. Ser. | 20 Jun 20 |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  | 1 |
| A3T-GCN | arXiv | 20 Jun 20 |  |  |  |  |  |  |  |  | 1 | 1 |  |  |  |  | 2 |
| TSE-SC | Trans-GIS | 1 Jun 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| MTGNN | arXiv | 24 May 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| ST-MetaNet+ | TKDE | 19 May 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| STGNN | WWW | 20 Apr 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| STSeq2Seq | arXiv | 6 Apr 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| DSTGNN | arXiv | 12 Mar 20 |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  | 1 |
| RSTAG | IoT-J | 19 Feb 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| GMAN | AAAI | 7 Feb 20 |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  | 1 |
| MRA-BGCN | AAAI | 7 Feb 20 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| STSGCN | AAAI | 7 Feb 20 |  |  |  |  |  | 1 | 1 |  |  |  | 1 | 1 |  |  | 4 |
| SLCNN | AAAI | 7 Feb 20 |  | 1 | 1 | 1 |  |  |  |  |  |  |  |  |  |  | 3 |
| DDP-GCN | arXiv | 7 Feb 20 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| R-SSM | ICLR | 13 Jan 20 |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  | 1 |
| GWNV2 | arXiv | 11 Dec 19 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| DeepGLO | NeurIPS | 8 Dec 19 | 1 |  |  | 1 |  |  |  |  |  |  |  |  |  |  | 1 |
| STGRAT | arXiv | 29 Nov 19 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| TGC-LSTM | T-ITS | 28 Nov 19 |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  | 1 |
| DCRNN-RIL | TrustCom/BigDataSE | 31 Oct 19 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| L-VGAE | arXiv | 18 Oct 19 |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  | 1 |
| T-GCN | T-ITS | 22 Aug 19 |  |  |  |  |  |  |  |  | 1 | 1 |  |  |  |  | 2 |
| GWN | IJCAI | 10 Aug 19 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| ST-MetaNet | KDD | 25 Jul 19 |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  | 1 |
| MRes-RGNN-G | AAAI | 17 Jul 19 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| CDSA | arXiv | 23 May 19 |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  | 1 |
| STDGI | ICLR | 12 Apr 19 |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  | 1 |
| ST-UNet | arXiv | 13 Mar 19 |  | 1 |  | 1 | 1 |  |  |  |  |  |  |  |  |  | 3 |
| 3D-TGCN | arXiv | 3 Mar 19 |  |  | 1 | 1 | 1 |  |  |  |  |  |  |  |  |  | 3 |
| ASTGCN | AAAI | 27 Jan 19 |  |  |  |  |  | 1 | 1 |  |  |  |  |  |  |  | 2 |
| PSN | T-ITS | 17 Aug 18 |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 0 |
| GaAN | UAI | 6 Aug 18 | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  | 1 |
| Seq2Seq Hybrid | KDD | 19 Jul 18 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| STGCN | IJCAI | 13 Jul 18 |  |  |  | 1 | 1 |  |  |  |  |  |  |  |  |  | 2 |
| DCRNN | ICLR | 30 Apr 18 |  | 1 | 1 |  |  |  |  |  |  |  |  |  |  |  | 2 |
| SBU-LSTM | UrbComp | 14 Aug 17 |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  | 1 |
| GRU | YAC | 5 Jan 17 |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  | 1 |







## Performance


![METR-LA MAE@60 mins](https://github.com/aprbw/traffic_prediction/blob/master/METRLA_MAE_60.PNG)

![PeMS-BAY MAE@60 mins](https://github.com/aprbw/traffic_prediction/blob/master/PeMSBAY_MAE_60.PNG)


NOTES: The experimental setttings may vary. But the common setting is:

* Observation window = 12 timesteps

* Prediction horizon = 1 timesteps

* Prediction window = 12 timesteps

* Metrics = MAE, RMSE, MAPE

* Train, validation, and test splits = 7/1/2 OR 6/2/2

However, there are many caveats:

* Some use different models for different prediction horizon.

* Some use different batch size when testing previous models, as they increase the observation and prediction windows from previous studies, and have difficulties fitting it on GPU using the same batch size.

* Regarding adjacency matrix, some derive it using Gaussian RBF from the coordinates, some use the actual connectivity, some simply learn it, and some use combinations.

* Some might also add more context, such as time of day, or day of the week, or weather.

* DeepGLO in particular, since it is treating it as a multi-channel timeseries without the spatial information, use rolling validation,

* Many different treatment of missing datasets, from exclusion to imputations.








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
    [Baidu with code: "p72z"](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw);
    [From Davidham3 Github STSGCN](https://github.com/Davidham3/STSGCN)

    * **PeMS-08** 
    [ATSGCN github](https://github.com/Davidham3/ASTGCN/tree/master/data);
    [Baidu with code: "p72z"](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw);
    [From Davidham3 github STSGCN](https://github.com/Davidham3/STSGCN)

    * **PeMS-03** 
    [Baidu with code: "p72z"](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw);
    [From Davidham3 github STSGCN](https://github.com/Davidham3/STSGCN)

    * **PeMS-07** 
    [Baidu with code: "p72z"](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw);
    [From Davidham3 github STSGCN](https://github.com/Davidham3/STSGCN)

    * **PeMS-SF** [UCI](https://archive.ics.uci.edu/ml/datasets/PEMS-SF)

    * **PeMS-11160** [GP-DCRNN github](https://github.com/tanwimallick/graph_partition_based_DCRNN)

* **LOOP** https://github.com/zhiyongc/Seattle-Loop-Data

* **Q-Traffic** https://github.com/JingqingZ/BaiduTraffic

[Baidu, code: 'umqd'](https://pan.baidu.com/share/init?surl=s1bauEJs8ONtC65ZkC4N3A)

* **RMTMC - MnDOT** https://www.d.umn.edu/~tkwon/TMCdata/TMCarchive.html The data in this archive are continuously collected by the Regional Trasportation Management Center (RTMC), a division of Minesotta Deaprtment of Transport (MnDOT) USA.

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

    * NE-BJ https://github.com/tsinghua-fib-lab/Traffic-Benchmark

* Xiamen https://ieeexplore.ieee.org/document/8029849

Also relevant:

* [Davidham3 list](https://github.com/Davidham3/open-traffic-datasets)




## Libraries

* [LibCity](https://libcity.ai/) [GitHub](https://github.com/LibCity/Bigscity-LibCity)

* Tsinghua Fib Lab [GitHub](https://github.com/tsinghua-fib-lab/Traffic-Benchmark)

* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io)


## Paper

The papers are sorted alphabetically based on model name. The citations are based on Google scholar citation.

You can find the bibtex in traffic_prediction.bib (not complete yet)






|      model     | citations |        venue        | published date | paper                                                                                                                                                                                                                                  | codes                                                      |
|:--------------:|:---------:|:-------------------:|:--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
|     3D-TGCN    |     12    |        arXiv        |    3 Mar 19    | [3D Graph Convolutional Networks with Temporal Graphs: A Spatial Information Free Framework For Traffic Forecasting](https://arxiv.org/abs/1903.00919)                                                                                 | []()                                                       |
|      AGCRN     |     3     |        arXiv        |    6 Jul 20    | [Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting](https://arxiv.org/abs/2007.02842)                                                                                                                             | [PyTorch](https://github.com/LeiBAI/AGCRN)                 |
|      ARNN      |     0     |     Nat. Commun     |    11 Sep 20   | [Autoreservoir computing for multistep ahead prediction based on the spatiotemporal information transformation](https://www.nature.com/articles/s41467-020-18381-0)                                                                    | []()                                                       |
|     ASTGCN     |     63    |         AAAI        |    27 Jan 19   | [Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting](https://www.aaai.org/ojs/index.php/AAAI/article/view/3881/3759)                                                                           | [Pytorch](https://github.com/guoshnBJTU/ASTGCN-r-pytorch) [MXNet](https://github.com/guoshnBJTU/ASTGCN) |
|      CDSA      |     2     |        arXiv        |    23 May 19   | [CDSA: Cross-Dimensional Self-Attention for Multivariate, Geo-tagged Time Series Imputation](https://arxiv.org/abs/1905.09904)                                                                                                         | []()                                                       |
|      CLGRN     |     0     |        arXiv        |     4 Jan 21   | [Conditional Local Filters with Explainers for Spatio-Temporal Forecasting](https://arxiv.org/abs/2101.01000)                                                                                                         | []()                                                       |
|      DCRNN     |    427    |         ICLR        |    30 Apr 18   | [DIFFUSION CONVOLUTIONAL RECURRENT NEURAL NETWORK: DATA-DRIVEN TRAFFIC FORECASTING](https://arxiv.org/abs/1707.01926v3)                                                                                                                | [tf](https://github.com/liyaguang/DCRNN) [PyTorch](https://github.com/chnsh/DCRNN_PyTorch)                  |
|    DCRNN-RIL   |     2     |  TrustCom/BigDataSE |    31 Oct 19   | [Diffusion Convolutional Recurrent Neural Network with Rank Influence Learning for Traffic Forecasting](https://ieeexplore.ieee.org/abstract/document/8887408)                                                                         | []()                                                       |
|     DDP-GCN    |     1     |        arXiv        |    7 Feb 20    | [DDP-GCN: Multi-Graph Convolutional Network for Spatiotemporal Traffic Forecasting](https://arxiv.org/abs/1905.12256)                                                                                                                  | []()                                                       |
|     DeepGLO    |     22    |       NeurIPS       |    8 Dec 19    | [Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting](https://arxiv.org/abs/1905.03806)                                                                                            | []()                                                       |
|      DGCRN     |     0     |        ArXiv        |   30 Apr 21    | [Spatiotemporal Adaptive Gated Graph Convolution Network for Urban Traffic Flow Forecasting](https://dl.acm.org/doi/abs/10.1145/3340531.3411894) | [PyTorch](https://github.com/RobinLu1209/STAG-GCN) |
|      DKFN      |      0    |      SIGSPATIAL     |    3 Nov 20    | [Graph Convolutional Networks with Kalman Filtering for Traffic Prediction](https://dl.acm.org/doi/10.1145/3397536.3422257)                                                                              | [PyTorch](https://github.com/Fanglanc/DKFN)                  |
|     DSTGNN     |     0     |        arXiv        |    12 Mar 20   | [Dynamic Spatiotemporal Graph Neural Network with Tensor Network](https://arxiv.org/abs/2003.08729)                                                                                                                                    | []()                                                       |
|     FASTGNN    |     0     |         TII         |    29 Jan 21   | [FASTGNN: A Topological Information Protected Federated Learning Approach For Traffic Speed Forecasting](https://ieeexplore.ieee.org/abstract/document/9340313)                                                                                                                                    | []()                                                       |
|     FC-GAGA    |     0     |        arXiv        |    30 Jul 20   | [FC-GAGA: Fully Connected Gated Graph Architecture for Spatio-Temporal Traffic Forecasting](https://arxiv.org/abs/2007.15531)                                                                                                          | []()                                                       |
|     FreqST     |     2     |        ICDM         |    17 Nov 21   | [FreqST: Exploiting Frequency Information in Spatiotemporal Modeling for Traffic Prediction](https://ieeexplore.ieee.org/abstract/document/9338305) | []() |
|      GaAN      |    126    |         UAI         |    6 Aug 18    | [GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs](http://auai.org/uai2018/proceedings/papers/139.pdf)                                                                                                   | [MXNet](https://github.com/jennyzhang0215/GaAN)            |
|      GDRNN     |     0     |         NTU         |    16 Aug 20   | [Deep learning approaches for traffic prediction](https://dr.ntu.edu.sg/handle/10356/142029)                                                                                                                                           | []()                                                       |
|      GMAN      |     20    |         AAAI        |    7 Feb 20    | [GMAN: A Graph Multi-Attention Network for Traffic Prediction](https://arxiv.org/abs/1911.08415)                                                                                                                                       | [tf](https://github.com/zhengchuanpan/GMAN)                |
|       GRU      |    308    |         YAC         |    5 Jan 17    | [Using LSTM and GRU neural network methods for traffic flow prediction](https://ieeexplore.ieee.org/abstract/document/7804912)                                                                                                         | [Keras](https://github.com/xiaochus/TrafficFlowPrediction) |
|     GST-GAT    |     0     |     IEEE Access      |    6 Jan 21    | [Modeling Global Spatial–Temporal Graph Attention Network for Traffic Prediction](https://ieeexplore.ieee.org/abstract/document/9316302) |                                                     |
|     G-SWaN    |     0     |     IoTDI      |    9 May 23    | [Because Every Sensor Is Unique, so Is Every Pair: Handling Dynamicity in Traffic Forecasting](https://arxiv.org/abs/2302.09956) |  [PyTorch](https://github.com/aprbw/G-SWaN)  |
|       GTS      |     0     |        UCONN        |    3 Aug 20    | [End-to-End Structure-Aware Convolutional Networks on Graphs](https://opencommons.uconn.edu/dissertations/2555/)                                                                                                                       | []()                                                       |
|       GTS      |     0     |        ICLR         |    4 May 21    | [Discrete Graph Structure Learning for Forecasting Multiple Time Series](https://openreview.net/forum?id=WEHSlH5mOk)       | [PyTorch](https://github.com/chaoshangcs/GTS) |
|       GWN      |     46    |        IJCAI        |    10 Aug 19   | [Graph WaveNet for Deep Spatial-Temporal Graph Modeling](https://www.ijcai.org/Proceedings/2019/0264.pdf)                                                                                                                              | [PyTorch](https://github.com/nnzhan/Graph-WaveNet)         |
|    GWNN-LSTM   |     0     | 	J. Phys. Conf. Ser. |    20 Jun 20   | [Graph Wavelet Long Short-Term Memory Neural Network: A Novel Spatial-Temporal Network for Traffic Prediction.](https://iopscience.iop.org/article/10.1088/1742-6596/1549/4/042070/meta)                                               | []()                                                       |
|      GWNV2     |     0     |        arXiv        |    11 Dec 19   | [Incrementally Improving Graph WaveNet Performance on Traffic Prediction](https://arxiv.org/abs/1912.07390)                                                                                                                            | [PyTorch](https://github.com/sshleifer/Graph-WaveNet)      |
|     H-STGCN    |     0     |         KDD         |    23 Aug 20   | [Hybrid Spatio-Temporal Graph Convolutional Network: Improving Traffic Prediction with Navigation Data](https://dl.acm.org/doi/pdf/10.1145/3394486.3403358)                                                                            | []()                                                       |
|     HetGAT     |     0     |        JAIHC        |    23 Jan 21   | [HetGAT: a heterogeneous graph attention network for freeway traffic speed prediction](https://link.springer.com/article/10.1007/s12652-020-02807-0)                                                                            | []()                                                       |
|    ISTD-GCN    |     0     |        arXiv        |    10 Aug 20   | [ISTD-GCN: Iterative Spatial-Temporal Diffusion Graph Convolutional Network for Traffic Speed Forecasting](https://arxiv.org/abs/2008.03970)                                                                                           | []()                                                       |
|     L-VGAE     |     0     |        arXiv        |    18 Oct 19   | [Decoupling feature propagation from the design of graph auto-encoders](https://arxiv.org/abs/1910.08589)                                                                                                                              | []()                                                       |
|      LSTM      |     39    |        TENCON       |    22 Nov 16   | [Traffic flow prediction with Long Short-Term Memory Networks (LSTMs)](https://ieeexplore.ieee.org/abstract/document/7848593)                                                                                                          | []()                                                       |
|       M2       |     1     |       J. AdHoc      |    1 Sep 20    | [A performance modeling and analysis of a novel vehicular traffic flow prediction system using a hybrid machine learning-based model](https://www.sciencedirect.com/science/article/abs/pii/S1570870520301803)                         | []()                                                       |
|    MRA-BGCN    |     7     |         AAAI        |    7 Feb 20    | [Multi-Range Attentive Bicomponent Graph Convolutional Network for Traffic Forecasting](https://aaai.org/ojs/index.php/AAAI/article/view/5758/5614)                                                                                    | []()                                                       |
|   MRes-RGNN-G  |     20    |         AAAI        |    17 Jul 19   | [Gated Residual Recurrent Graph Neural Networks for Traffic Prediction](https://www.aaai.org/ojs/index.php/AAAI/article/view/3821)                                                                                                     | []()                                                       |
|      MTGNN     |     7     |        arXiv        |    24 May 20   | [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/abs/2005.11650)                                                                                                               | []()                                                       |
|       PSN      |     4     |        T-ITS        |    17 Aug 18   | [Pattern Sensitive Prediction of Traffic Flow Based on Generative Adversarial Framework](https://ieeexplore.ieee.org/document/8438991)                                                                                                 | []()                                                       |
|      R-SSM     |     0     |         ICLR        |    13 Jan 20   | [Relational State-Space Model for Stochastic Multi-Object Systems](https://arxiv.org/abs/2001.04050)                                                                                                                                   | []()                                                       |
|      RSTAG     |     3     |        IoT-J        |    19 Feb 20   | [Reinforced Spatiotemporal Attentive Graph Neural Networks for Traffic Forecasting](https://ieeexplore.ieee.org/abstract/document/9003261)                                                                                             | []()                                                       |
|       SAE      |    1626   |        T-ITS        |    9 Sep 14    | [Traffic flow prediction with big data: a deep learning approach](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6894591&casa_token=5sdFl519NNoAAAAA:iBT5EdQxzJPUr_Ljh3nT1nM83jCux71OKcG7RUrctrMi1sgPb49Sb1GLU_CGZ9AA92w9y-B-vg) | [Keras](https://github.com/xiaochus/TrafficFlowPrediction) |
|    SBU-LSTM    |    157    |       UrbComp       |    14 Aug 17   | [Deep Bidirectional and Unidirectional LSTM Recurrent Neural Network for Network-wide Traffic Speed Prediction](https://arxiv.org/abs/1801.02143)                                                                                      | []()                                                       |
|    SCPT    |    0    |       ArXiv       |    9 Aug 23   | [Traffic Forecasting on New Roads Unseen in the Training Data Using Spatial Contrastive Pre-Training](https://arxiv.org/abs/2305.05237)                                                                                      | []()                                                       |
| Seq2Seq Hybrid |     48    |         KDD         |    19 Jul 18   | [Deep Sequence Learning with Auxiliary Information for Traffic Prediction](https://dl.acm.org/doi/10.1145/3219819.3219895)                                                                                                             | [tf](https://github.com/JingqingZ/BaiduTraffic)            |
|      SGMN      |     1     |        J. TRC       |    20 Aug 20   | [Graph Markov network for traffic forecasting with missing data](https://www.sciencedirect.com/science/article/pii/S0968090X20305866)                                                                                                  | []()                                                       |
|      SLCNN     |     1     |         AAAI        |    7 Feb 20    | [Spatio-Temporal Graph Structure Learning for Traffic Forecasting](https://aaai.org/ojs/index.php/AAAI/article/view/5470/5326)                                                                                                         | []()                                                       |
|   ST-MetaNet   |     39    |         KDD         |    25 Jul 19   | [Urban traffic prediction from spatio-temporal data using deep meta learning](https://dl.acm.org/doi/pdf/10.1145/3292500.3330884)  | [MXNet](https://github.com/panzheyi/ST-MetaNet)                                                       |
|   ST-MetaNet+  |     0     |         TKDE        |    19 May 20   | [Spatio-Temporal Meta Learning for Urban Traffic Prediction](https://ieeexplore.ieee.org/abstract/document/9096591)                                                                                                                    | []()                                                       |
|  ST-TrafficNet |     2     |        ELECGJ       |    9 Sep 20    | [ST-TrafficNet: A Spatial-Temporal Deep Learning Network for Traffic Forecasting](https://www.mdpi.com/2079-9292/9/9/1474/htm)                                                                                                         | []()                                                       |
|     ST-UNet    |     11    |        arXiv        |    13 Mar 19   | [ST-UNet: A Spatio-Temporal U-Network for Graph-structured Time Series Modeling](https://arxiv.org/abs/1903.05631)                                                                                                                     | []()                                                       |
|      STDGI     |     3     |         ICLR        |    12 Apr 19   | [Spatio-Temporal Deep Graph Infomax](https://arxiv.org/abs/1904.06316)                                                                                                                                                                 | []()                                                       |
|      STGAT     |     0     |     IEEE Access     |    22 Jul 20   | [STGAT: Spatial-Temporal Graph Attention Networks for Traffic Flow Forecasting](https://ieeexplore.ieee.org/abstract/document/9146162)                                                                                                 | []()                                                       |
|      STGAT     |     0     |     IEEE Access     |    22 Jul 20   | [STGAT: Spatial-Temporal Graph Attention Networks for Traffic Flow Forecasting](https://ieeexplore.ieee.org/abstract/document/9146162)                                                                                                 | []()                                                       |
|      STGCN     |    322    |        IJCAI        |    13 Jul 18   | [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875)                                                                                                    | [tf](https://github.com/PKUAI26/STGCN-IJCAI-18) [PyTorch](https://github.com/FelixOpolka/STGCN-PyTorch) [MXNet](https://github.com/Davidham3/STGCN)           |
|      STGNN     |     4     |         WWW         |    20 Apr 20   | [Traffic Flow Prediction via Spatial Temporal Graph Neural Network](https://dl.acm.org/doi/abs/10.1145/3366423.3380186)                                                                                                                | []()                                                       |
|     STGRAT     |     6     |        arXiv        |    29 Nov 19   | [STGRAT: A Spatio-Temporal Graph Attention Network for Traffic Forecasting](https://arxiv.org/abs/1911.13181)                                                                                                                          | []()                                                       |
|      STNN      |     0     |        T-ITS        |    16 Jul 20   | [STNN: A Spatio-Temporal Neural Network for Traffic Predictions](https://ieeexplore.ieee.org/document/9142387)                                                                                                                         | []()                                                       |
|    STSeq2Seq   |     0     |        arXiv        |    6 Apr 20    | [Forecast Network-Wide Traffic States for Multiple Steps Ahead: A Deep Learning Approach Considering Dynamic Non-Local Spatial Correlation and Non-Stationary Temporal Dependency](https://arxiv.org/abs/2004.02391)                   | []()                                                       |
|      STGAM     |     1     |      CISP-BMEI      |   17 Oct 20    | [Spatial-Temporal Graph Attention Model on Traffic Forecasting](https://ieeexplore.ieee.org/document/9263680)| []()                                                       |
|     STSGCN     |     5     |         AAAI        |    7 Feb 20    | [Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting](https://github.com/Davidham3/STSGCN/blob/master/paper/AAAI2020-STSGCN.pdf)                                  | [MXNet](https://github.com/Davidham3/STSGCN)               |
|    TGC-LSTM    |     95    |        T-ITS        |    28 Nov 19   | [Traffic Graph Convolutional Recurrent Neural Network: A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting](https://ieeexplore.ieee.org/document/8917706)                                                     | [](https://github.com/zhiyongc/Graph_Convolutional_LSTM)   |
|     TSE-SC     |     0     |      Trans-GIS      |    1 Jun 20    | [Traffic transformer: Capturing the continuity and periodicity of time series for traffic forecasting](https://onlinelibrary.wiley.com/doi/pdf/10.1111/tgis.12644)                                                                     | []()                                                       |
|     TSSRGCN    |     4     |        ICDM         |    17 Nov 21    | [TSSRGCN: Temporal Spectral Spatial Retrieval Graph Convolutional Network for Traffic Flow Forecasting](https://ieeexplore.ieee.org/abstract/document/9338393) | []()                                                       |
|                |     0     |        arXiv        |    15 Jul 20   | [On the Inclusion of Spatial Information for Spatio-Temporal Neural Networks](https://arxiv.org/abs/2007.07559)                                                                                                                        | [PyTorch](https://github.com/rdemedrano/SANN)              |
|                |     96    |        NeuCom       |    27 Nov 18   | [LSTM-based traffic flow prediction with missing data](https://www.sciencedirect.com/science/article/abs/pii/S0925231218310294)                                                                                                        | []()                                                       |





Things that would be in the table above if I have more time:











## Other works


* Multi-Attention Temporal and Graph Convolution Network for Traffic Flow Forecasting [PyTorch](https://github.com/lk485/matgcn)

* [Foreseeing Congestion using LSTM on Urban Traffic Flow Clusters](https://ieeexplore.ieee.org/document/9010150)
ICSAI 2019
[Keras](https://github.com/wangz315/ClusterPredictTrafficFlow);
dataset: CityPulse

* [Using LSTM and GRU neural network methods for traffic flow prediction](https://ieeexplore.ieee.org/abstract/document/7804912)
IEEE YAC 2016
[Keras](https://github.com/xiaochus/TrafficFlowPrediction);
dataset: PeMS but different from everyone else

* [A Dynamic Traffic Awareness System for Urban Driving](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8875288)
IEEE GreenCom 2019
[Keras](https://github.com/wangz315/ClusterPredictTrafficFlow);
dataset: CityPulse

* [Inductive Graph Neural Networks for Spatiotemporal Kriging (IGNNK)](https://arxiv.org/abs/2006.07527)
AAAI 2021
[PyTorch](https://github.com/Kaimaoge/IGNNK)
dataset: METR-LA, PeMS-BAY, LOOP, NREL, USHCN.




Other works that is not based on a static-spatial-graph of timeseries:

* [VLUC](https://arxiv.org/abs/1911.06982)

* [STDN](https://arxiv.org/pdf/1803.01254.pdf)

* [Deep Representation Learning for Trajectory Similarity Computation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8509283)

* [Curb-GAN](https://github.com/Curb-GAN/Curb-GAN) SIG KDD 2020

* [BusTr](https://dl.acm.org/doi/pdf/10.1145/3394486.3403376) SIG KDD 2020

* [DeepMove](https://dl.acm.org/doi/pdf/10.1145/3178876.3186058) WWW 2018

* https://github.com/Alro10/deep-learning-time-series

* https://github.com/henriquejosefaria/CSC

* https://github.com/shakibyzn/Traffic-flow-prediction

* https://deepmind.com/blog/article/traffic-prediction-with-advanced-graph-neural-networks








Other lists: 

* [Davidham3 list](https://github.com/Davidham3/spatio-temporal-paper-list)

* https://paperswithcode.com/task/traffic-prediction

* [A Survey on Modern Deep Neural Network for Traffic Prediction: Trends, Methods and Challenges](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9112608) IEEE TKDE 2020

* [A Comprehensive Survey on Traffic Prediction](https://arxiv.org/abs/2004.08555) to be published in IEEE Transactions on Intelligent Transportation Systems.

* [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/pdf/1901.00596.pdf) IEEE Trans. Neural Netw. Learn. Syst. 2020

* [A Comprehensive Survey on Geometric Deep Learning](https://ieeexplore.ieee.org/abstract/document/9003285) IEEE Access 19 Feb 2020.

* [Graph Neural Network for Traffic Forecasting: A Survey](https://arxiv.org/abs/2101.11174) Expert Systems with Applications [GitHub](https://github.com/jwwthu/GNN4Traffic)

* [BigsCity LibCity: Traffic Prediction Paper Collection](https://github.com/LibCity/Bigscity-LibCity-Paper) In the paper collection, we collected traffic prediction papers published in the recent years (2016-now) on 11 top conferences and journals, namely, AAAI, IJCAI, KDD, CIKM, ICDM, WWW, NIPS, ICLR, SIGSPATIAL, IEEE TKDE and IEEE TITS. In addition, the surveys since 2016 and representative papers mentioned in the surveys are also included. 

* Older, pre-ML approaches: [On the modeling of traffic and crowds: A survey of models, speculations, and perspectives](https://epubs.siam.org/doi/pdf/10.1137/090746677?casa_token=ramTXeUx3owAAAAA%3ABIA7wFjs6ZdGWqqCQ2iicLrZfUaZSTgZJtO8eYGDSvaI5IFPIQkCoZPi_btisCGJEV43HDedswY&) SIAM 2011

* https://github.com/Knowledge-Precipitation-Tribe/Urban-computing-papers

* https://github.com/jdlc105/Must-read-papers-and-continuous-tracking-on-Graph-Neural-Network-GNN-progress

* [DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction](https://arxiv.org/abs/2108.09091) CIKM 2021 [GitHub PyTorch](https://github.com/deepkashiwa20/DL-Traff-Grid)























# Acknowledgement

* [Willianto Aslim](https://github.com/asalimw)
