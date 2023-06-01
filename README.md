# DeepNCSPP
DeepNCSPP: A non-classical secreted protein prediction model based on deep learning

# Requirement
Python==3.8.16<br>
Pytorch-GPU==2.0.0<br>
Numpy==1.24.3<br>
cuda version 11.7<br>
cudnn version 8.0<br>
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# train cross validation 
python train.py

# train test dataset
python train_test.py

# demo
To facilitate testing, we put a trained weight parameter file in the demo folder demo/model.params<br>
python demo.py

# Note
If you want to train and get the same experimental results, please use the same operating system Windows 10 and the packages in requirements


