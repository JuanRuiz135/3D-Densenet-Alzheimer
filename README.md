# 3D Densenet in 4-way Classification of Alzheimer's Disease

Paper published in the Brain Informatics 2020 peer-reviewed journal.
<br/>
[
Ruiz J., Mahmud M., Modasshir M., Shamim Kaiser M., Alzheimer’s Disease Neuroimaging Initiative, <br/>
"3D DenseNet Ensemble in 4-Way Classification of Alzheimer’s Disease", <br/>
 Brain Informatics. BI 2020. 
](https://doi.org/10.1007/978-3-030-59277-6_8)

## Dataset and weights
The dataset and the model weights can be downloaded from this [drive](https://drive.google.com/drive/folders/12WrBiJb0qZ-u75nZACgbRLx_4HYIvYS8?usp=sharing).
<br/>
Both NiFTiFiles and model_weights folders have to be on the same folder as the python code.
<br/>
### Folder structure
    .
    ├── ...
    ├── cnn_interpretability        
    │   └── utils.py
    ├── model_weights           
    │   ├── classifier1_weights.pth         
    │   ├── classifier2_weights.pth    
    │   └── classifier3_weights.pth
    ├── NiFTiFiles 
    │   └── all the .nii files ...
    ├── model1.py
    ├── model2.py
    ├── model3.py
    └── ensemble.py
## Dependencies
Developed using CUDA 10.1 with cudnn 8.0.2
- [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-update2)
- [cudnn 8.0.2](https://developer.nvidia.com/rdp/cudnn-download)
### Libraries
- pytorch 1.6.0
- nibabel 3.1.1
- GitPython 3.1.7
- kaggle 1.5.8
- pandas 1.1.0
- numpy 1.19.1
- matplotlib 3.3.1
- The utils.py file was taken from this Github https://github.com/jrieke/cnn-interpretability. 
