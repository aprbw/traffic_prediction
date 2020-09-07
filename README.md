# Traffic Prediction

Traffic prediction is the task of predicting future traffic measurements (e.g. volume, speed, etc.) in a road network (graph), using historical data (timeseries).

Also relevant: https://paperswithcode.com/task/traffic-prediction

The papers are haphazardly selected.

## Summary

A tabular summary of paper and dataset.
The paper is reverse chronologically sorted.
The dataset is first sorted by if it is publically available
(A = publically **A**vailable;
N = **N**ot publically available),
and then number of usage.



|   model  |  venue  | published date |    A    |     A    |      A     |      A     |    A    |    A    |   A  |    A    |    A    |   N   |   N   |  N  |  N  |   N   |    N   |       |
|:--------:|:-------:|:--------------:|:-------:|:--------:|:----------:|:----------:|:-------:|:-------:|:----:|:-------:|:-------:|:-----:|:-----:|:---:|:---:|:-----:|:------:|-------|
|          |         |                | METR-LA | PeMS-BAY | PeMS-D7(M) | PeMS-D7(L) | PeMS-04 | PeMS-08 | LOOP | PeMS-03 | PeMS-07 | INRIX | BJER4 | BJF | BRF | BRF-L | Xiamen | TOTAL |
|   AGCRN  |  arXiv  |   6 Jul 2020   |         |          |            |            |    1    |    1    |      |         |         |       |       |     |     |       |        |   2   |
|   GMAN   |   AAAI  |   7 Feb 2020   |         |     1    |            |            |         |         |      |         |         |       |       |     |     |       |    1   |   2   |
| MRA-BGCN |   AAAI  |   7 Feb 2020   |    1    |     1    |            |            |         |         |      |         |         |       |       |     |     |       |        |   2   |
|  STSGCN  |   AAAI  |   7 Feb 2020   |         |          |            |            |    1    |    1    |      |    1    |    1    |       |       |     |     |       |        |   4   |
|   SLCNN  |   AAAI  |   7 Feb 2020   |    1    |     1    |      1     |            |         |         |      |         |         |       |       |  1  |  1  |   1   |        |   6   |
|   GWNV2  |  arXiv  |   11 Dec 2019  |    1    |     1    |            |            |         |         |      |         |         |       |       |     |     |       |        |   2   |
| TGC-LSTM |  T-ITS  |   28 Nov 2019  |         |          |            |            |         |         |   1  |         |         |   1   |       |     |     |       |        |   2   |
|    GWN   |  IJCAI  |   10 Aug 2019  |    1    |     1    |            |            |         |         |      |         |         |       |       |     |     |       |        |   2   |
|  ST-UNet |  arXiv  |    13 Mar 19   |    1    |          |      1     |      1     |         |         |      |         |         |       |       |     |     |       |        |   3   |
|  3D-TGCN |  arXiv  |   3 Mar 2019   |         |          |      1     |      1     |         |         |      |         |         |       |       |     |     |       |        |   2   |
|  ASTGCN  |   AAAI  |   27 Jan 2019  |         |          |            |            |    1    |    1    |      |         |         |       |       |     |     |       |        |   2   |
|   STGCN  |  IJCAI  |    13 Jul 18   |         |          |      1     |      1     |         |         |      |         |         |       |   1   |     |     |       |        |   3   |
|   DCRNN  |   ICLR  |    30 Apr 18   |    1    |     1    |            |            |         |         |      |         |         |       |       |     |     |       |        |   2   |
| SBU-LSTM | UrbComp |   14 Aug 2017  |         |          |            |            |         |         |   1  |         |         |   1   |       |     |     |       |        |   2   |
|          |         |      TOTAL     |    6    |     6    |      4     |      3     |    3    |    3    |   2  |    1    |    1    |   2   |   1   |  1  |  1  |   1   |    1   |       |









## Performance

NOTE: The experimental setttings may vary. But the common setting is:

* Observation window = 12 timesteps

* Prediction horizon = 1 timesteps

* Prediction window = 12 timesteps









## Dataset

METR-LA	PeMS-BAY	PeMS-D7(M)	PeMS-D7(L)	LOOP	PeMS-04	PeMS-08	PeMS03	PeMS04	PeMS07	PeMS08

The following datasets are not publically available:

* INRIX

* BJER4

* BJF

* BRF

* BRF-L

* Xiamen








## Paper

The papers are sorted alphabetically. The citations are based on Google scholar citation on 2020 09 07.


| model    | citations | venue   | published date | paper                                                                                                                              | link                                                                                                                                                                      |
|----------|-----------|---------|----------------|------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3D-TGCN  |        12 | arXiv   |     3 Mar 2019 | 3D Graph Convolutional Networks with Temporal Graphs: A Spatial Information Free Framework For Traffic Forecasting                 | https://arxiv.org/abs/1903.00919                                                                                                                                          |
| AGCRN    |         3 | arXiv   |     6 Jul 2020 | Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting                                                             | https://arxiv.org/abs/2007.02842                                                                                                                                          |
| ASTGCN   |        63 | AAAI    |    27 Jan 2019 | Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting                                         | https://www.aaai.org/ojs/index.php/AAAI/article/view/3881/3759                                                                                                            |
| SBU-LSTM |       157 | UrbComp |    14 Aug 2017 | Deep Bidirectional and Unidirectional LSTM Recurrent Neural Network for Network-wide Traffic Speed Prediction                      | https://arxiv.org/abs/1801.02143                                                                                                                                          |
| DCRNN    |       389 | ICLR    |      30 Apr 18 | DIFFUSION CONVOLUTIONAL RECURRENT NEURAL NETWORK: DATA-DRIVEN TRAFFIC FORECASTING                                                  | https://arxiv.org/abs/1707.01926v3                                                                                                                                        |
| GMAN     |        20 | AAAI    |     7 Feb 2020 | GMAN: A Graph Multi-Attention Network for Traffic Prediction                                                                       | https://arxiv.org/abs/1911.08415                                                                                                                                          |
| GWN      |        46 | IJCAI   |    10 Aug 2019 | Graph WaveNet for Deep Spatial-Temporal Graph Modeling                                                                             | https://www.ijcai.org/Proceedings/2019/0264.pdf                                                                                                                           |
| GWNV2    |         0 | arXiv   |    11 Dec 2019 | Incrementally Improving Graph WaveNet Performance on Traffic Prediction                                                            | https://arxiv.org/abs/1912.07390                                                                                                                                          |
| MRA-BGCN |         7 | AAAI    |     7 Feb 2020 | Multi-Range Attentive Bicomponent Graph Convolutional Network for Traffic Forecasting                                              | https://aaai.org/ojs/index.php/AAAI/article/view/5758/5614                                                                                                                |
| STSGCN   |         5 | AAAI    |     7 Feb 2020 | Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting           | https://github.com/Davidham3/STSGCN/blob/master/paper/AAAI2020-STSGCN.pdf                                                                                                 |
| STGCN    |       322 | IJCAI   |      13 Jul 18 | Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting                                    | https://arxiv.org/abs/1709.04875                                                                                                                                          |
| SLCNN    |         1 | AAAI    |     7 Feb 2020 | Spatio-Temporal Graph Structure Learning for Traffic Forecasting                                                                   | https://aaai.org/ojs/index.php/AAAI/article/view/5470/5326                                                                                                                |
| ST-UNet  |        11 | arXiv   |      13 Mar 19 | ST-UNet: A Spatio-Temporal U-Network for Graph-structured Time Series Modeling                                                     | https://arxiv.org/abs/1903.05631                                                                                                                                          |
| TGC-LSTM |        95 | T-ITS   |    28 Nov 2019 | Traffic Graph Convolutional Recurrent Neural Network: A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting | https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8917706&casa_token=Vfm5kpRcwiwAAAAA:B74LVeC_MpayC0Z_CwOXRPopf13f3qosTy2SxhaDqchCNG3SBMJ8qfpmkYCwfO4R0hI-bSJ4XQ&tag=1 |


