# Traffic Prediction

Traffic prediction is the task of predicting future traffic measurements (e.g. volume, speed, etc.) in a road network (graph), using historical data (timeseries).

Also relevant: https://paperswithcode.com/task/traffic-prediction

The papers are haphazardly selected.

## Summary

A tabular summary of paper and dataset.
The paper is reverse chronologically sorted.
The dataset is first sorted by if it is publically available
(A = publically **A**vailable
N = **N**ot publically available),
and then number of usage.



|   model  |  venue  | published date |    A    |     A    |      A     |      A     |   A  |    A    |    A    |    A   |    A   |    A   |    A   |   N   |   N   |  N  |  N  |   N   |    N   |       |
|:--------:|:-------:|:--------------:|:-------:|:--------:|:----------:|:----------:|:----:|:-------:|:-------:|:------:|:------:|:------:|:------:|:-----:|:-----:|:---:|:---:|:-----:|:------:|-------|
|          |         |                | METR-LA | PeMS-BAY | PeMS-D7(M) | PeMS-D7(L) | LOOP | PeMS-04 | PeMS-08 | PeMS03 | PeMS04 | PeMS07 | PeMS08 | INRIX | BJER4 | BJF | BRF | BRF-L | Xiamen | TOTAL |
|   AGCRN  |  arXiv  |   6 Jul 2020   |         |          |            |            |      |    1    |    1    |        |        |        |        |       |       |     |     |       |        |   2   |
|   GMAN   |   AAAI  |   7 Feb 2020   |         |     1    |            |            |      |         |         |        |        |        |        |       |       |     |     |       |    1   |   2   |
| MRA-BGCN |   AAAI  |   7 Feb 2020   |    1    |     1    |            |            |      |         |         |        |        |        |        |       |       |     |     |       |        |   2   |
|  STSGCN  |   AAAI  |   7 Feb 2020   |         |          |            |            |      |         |         |    1   |    1   |    1   |    1   |       |       |     |     |       |        |   4   |
|   SLCNN  |   AAAI  |   7 Feb 2020   |    1    |     1    |      1     |            |      |         |         |        |        |        |        |       |       |  1  |  1  |   1   |        |   6   |
|   GWNV2  |  arXiv  |   11 Dec 2019  |    1    |     1    |            |            |      |         |         |        |        |        |        |       |       |     |     |       |        |   2   |
| TGC-LSTM |  T-ITS  |   28 Nov 2019  |         |          |            |            |   1  |         |         |        |        |        |        |   1   |       |     |     |       |        |   2   |
|    GWN   |  IJCAI  |   10 Aug 2019  |    1    |     1    |            |            |      |         |         |        |        |        |        |       |       |     |     |       |        |   2   |
|  ST-UNet |  arXiv  |    13 Mar 19   |    1    |          |      1     |      1     |      |         |         |        |        |        |        |       |       |     |     |       |        |   3   |
|  3D-TGCN |  arXiv  |   3 Mar 2019   |         |          |      1     |      1     |      |         |         |        |        |        |        |       |       |     |     |       |        |   2   |
|  ASTGCN  |   AAAI  |   27 Jan 2019  |         |          |            |            |      |    1    |    1    |        |        |        |        |       |       |     |     |       |        |   2   |
|   STGCN  |  IJCAI  |    13 Jul 18   |         |          |      1     |      1     |      |         |         |        |        |        |        |       |   1   |     |     |       |        |   3   |
|   DCRNN  |   ICLR  |    30 Apr 18   |    1    |     1    |            |            |      |         |         |        |        |        |        |       |       |     |     |       |        |   2   |
| SBU-LSTM | UrbComp |   14 Aug 2017  |         |          |            |            |   1  |         |         |        |        |        |        |   1   |       |     |     |       |        |   2   |
|          |         |      TOTAL     |    6    |     6    |      4     |      3     |   2  |    2    |    2    |    1   |    1   |    1   |    1   |   2   |   1   |  1  |  1  |   1   |    1   |       |


## Performance

NOTE: The experimental setttings may vary. But the common setting is:

* Observation window = 12 timesteps

* Prediction horizon = 1 timesteps

* Prediction window = 12 timesteps

## Dataset



## Paper

* **AGCRN**	Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting
* **ASTGCN**	Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting
* **GMAN**	GMAN: A Graph Multi-Attention Network for Traffic Prediction
* **GWN**	Graph WaveNet for Deep Spatial-Temporal Graph Modeling
* **GWNV2**	Incrementally Improving Graph WaveNet Performance on Traffic Prediction
* **MRA-BGCN**	Multi-Range Attentive Bicomponent Graph Convolutional Network for Traffic Forecasting
* **STSGCN**	Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting
* **SLCNN**	Spatio-Temporal Graph Structure Learning for Traffic Forecasting
