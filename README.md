# HiHa: Hierarchical Harmonic Decomposition Implicit Neural Compression for Atmospheric Data
![image](https://github.com/xzwbsz/HiHa/assets/44642002/55bf81be-2c47-4607-902a-6f9790badcc5)

## Introduction
Implicit neural representation (INR) is an emerging technique that gains momentum and demonstrates high promise for fitting diverse natural data. We propose a Hierarchical Harmonic Decomposition Implicit Neural Compression (HiHa) for atmospheric data, leveraging atmospheric-level spherical coordinates basis. 

## Preparation
The most significant component is Siren. And other requrements are as following:
Cartopy                     0.23.0
cdsapi                      0.6.1
matplotlib                  3.8.0
matplotlib-inline           0.1.7
mdurl                       0.1.2
mkl-fft                     1.3.8
mkl-random                  1.2.4
mkl-service                 2.4.0
nest_asyncio                1.6.0
networkx                    3.2.1
numcodecs                   0.12.1
numpy                       1.26.3
oauthlib                    3.2.2
opencv-python               4.9.0.80
opencv-python-headless      4.9.0.80
opt-einsum                  3.3.0
optax                       0.2.2
orbax-checkpoint            0.5.10
packaging                   23.1
pandas                      2.2.1
pillow                      10.2.0
pip                         23.3.1
pyparsing                   3.0.9
pyproj                      3.6.1
PyQt5                       5.15.10
PyQt5-sip                   12.13.0
pyshp                       2.3.1
PySocks                     1.7.1
python-dateutil             2.8.2
pytz                        2024.1
PyYAML                      6.0.1
pyzmq                       26.0.2
rich                        13.7.1
rsa                         4.9
Rtree                       1.2.0
scikit-image                0.22.0
scikit-learn                1.4.2
scikit-video                1.1.11
scipy                       1.12.0
termcolor                   2.4.0
threadpoolctl               3.5.0
tifffile                    2024.2.12
tomli                       2.0.1
toolz                       0.12.1
torch                       2.2.1+cu121
torchaudio                  2.2.1+cu121
torchvision                 0.17.1+cu121
tqdm                        4.66.2
traitlets                   5.14.3
trimesh                     4.1.7
triton                      2.2.0
wcwidth                     0.2.13
wheel                       0.41.2
xarray                      2023.7.0
yarl                        1.9.4
zarr                        2.17.1

## Guidance
Users can modify the parameter in config.yaml for each Siren layer and frequency.

Users cann also custom the input and output dir in sim-chunk.py

To start compression, use 

```python
python sim-chunk.py
```

To uncompression, use 
```python
python uncompress.py
```

The code will output the comparison img as 'out_image.png'

## Results
![image](https://github.com/user-attachments/assets/bb75fc25-1ad3-46d9-927b-d9147d0fe93f)
![image](https://github.com/user-attachments/assets/120a3d6f-aee4-4817-8ac9-586f499ab5ab)


