# CoaDTI
Multi-modal co-attention for drug-target interaction annotation and Its Application to SARS-CoV-2  

## Abstract


## Environment
The test was conducted in the linux server with GTX2080Ti and the running environment is as follows:
* python 3.7
* pytorch 1.7.1
* rdkit 
* pytorch geometric 1.6.3
* Cuda 10.0.130
## Data
Human and c.elegans dataset are available at \url{https://github.com/masashitsubaki/CPI_prediction/tree/master/dataset}. Binding\_DB dataset is available at \url{https://github.com/IBM/InterpretableDTIP}. The data of SARS-CoV-2 Main protease (Mpro) in complex with GC373 is available at \url{https://www.rcsb.org/structure/6WTK}. The data of SARS-CoV-2 Main protease (Mpro) in complex with ML188  is avalable at \url{https://www.rcsb.org/structure/7L0D}.




## How to run
### CoaDTI
1. Run ./code/data_prepare.py to preprocess the dataset.
2. Run ./code/train.py to train the CoaDTI.
#### CoaDTI-pro
1. Run ./code/data_prepare.py to preprocess the dataset.
2. Run ./code/train.py to train the CoaDTI-pro.

