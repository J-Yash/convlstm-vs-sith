# convlstm-vs-sith
Implementation for ConvLSTM and SITH models on Moving MNIST and Something-Something dataset for future video frame prediction and action recognition.

The repository contains 4 jupyter notebooks:
1. **convlstm_pytorch_moving_mnist:** Contains PyTorch implementation of a ConvLSTM model on Moving MNIST dataset for future video frame prediction.
2. **SITH_pytorch_moving_mnist:** Contains PyTorch implementation of a SITH (log compressed memory) model on Moving MNIST dataset for future video frame prediction.
3. **convlstm_pytorch_something_something:** Contains PyTorch implementation of a ConvLSTM model on Something Something V2 dataset for video action recognition.
4. **SITH_pytorch_something_something:** Contains PyTorch implementation of a SITH (log compressed memory) model on Something Something V2 dataset for video action recognition.

Code references:
* **Moving MNIST:** https://github.com/tychovdo/MovingMNIST.git
* **ConvLSTM:** https://github.com/holmdk/Video-Prediction-using-PyTorch
* **SITH:** https://github.com/compmem/SITH_Layer
* **Something Something (Action Recognition) Baseline:** https://github.com/TwentyBN/something-something-v2-baseline

NOTE: Notebooks were run on Google Colab with Google Drive as data storage, hence the code is set up to be used that way.

Discussion:
1. **Something Something Dataset:** The something something v2 dataset is very large and contains over 200000 videos and 174 classes. Hence, it is very time-consuming to train the model on this dataset. Training just 1 epoch on the dataset takes hours and reducing the training data introduces unbalanced classes (if done without further preprocessing of data). Best option would be to use a multi-GPU setup and train the model in a distributed training method.
2. **Moving MNIST:** Training the models on this is also time-consuming, but manageable. Both ConvLSTM as well as SITH models were trained for 300 epochs (taking over 6 hours in each case). While we see better results for the ConvLSTM model, the SITH model fails to learn much. The reason for this might be because the SITH models are better when there is a temporal dependency across a longer period of time. In our case, we use 10 frames to predict the next 10 frames. This is suboptimal for SITH models. In order to better leverage the SITH model, it would be a better idea to use 1000 or more frames to predict the next 10 ( or more) frames. In such a case, SITH model can leverage the log compressed memory to better track the patterns across longer period of time. 
