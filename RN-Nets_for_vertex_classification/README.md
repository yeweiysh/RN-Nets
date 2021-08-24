# RN-Nets

## Overview
Here we provide the implementation of RN-Nets in TensorFlow. The repository is organised as follows:
- `data/` contains datasets Cora, Citeseer, and Pubmed;
- `new_data/` contains datasets Chameleon and Squirrel;
- `models/` contains the implementation of the RN-Nets (`rnnets.py`);
- `utils/` contains:
    * an implementation of the aggregation of each matrix in the Krylov tensor by a one-dim CNN (`layers.py`);
    * preprocessing subroutines (`process.py`);

Finally, `bash run_train` execute the experiments.


## Dependencies

The script has been tested running under Python 3.7.9, with TensorFlow version as:
- `tensorflow-gpu==1.13.1`

In addition, CUDA 10.2 and cuDNN 8.1 have been used.


## License
MIT
