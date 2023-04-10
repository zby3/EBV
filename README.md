<p>
   <img width="1000" src="figs/workflow.png"></a>
</p>
<br>

## Deep Learning Predicts EBV Status in Gastric Cancer based on Spatial Patterns of Lymphocyte Infiltration
_**Baoyi Zhang<sup>1</sup>**, Kevin Yao<sup>2</sup>, Min Xu<sup>3</sup>, Jia Wu<sup>4</sup>, Chao Cheng<sup>5,*</sup>_</br></br>
### Table of Contents  
[Requirements](#requirements)  
[Overview](#overview)  
[Preprocess](#preprocess)  
[Training and Test](#training)  


<a name="requirements"></a>
### Requirements

* python 3.7
* openslides >= 3.3
* torch >= 1.9.1

<a name="overview"></a>
### Overview

This repository provides the codes for training and test the EBV infection detection model in the above article. First, a tumor vs. normal model was trained to classify each images into tumor or normal. Then, EBV prediction models were trained seperately on tumor and normal images. 

<a name="preprocess"></a>
### Preprocess

Run the xxx.py to cut large whole slide images into small tiles with 512\*512 pixels. 
```
$ python xxx.py 
```

Run the xxx.py to perform color normalizarion of all tiles using xxx methods. 
```
$ python xxx.py 
```

<a name="training"></a>
### Training and Test
#### Tumor vs. normal model
Run the xxx.py to train the tumor vs. normal model
```
$ python xxx.py 
```

Run the xxx.py to test the tumor vs. normal model
```
$ python xxx.py 
```
#### EBV model
Run the xxx.py to train the EBV model
```
$ python xxx.py 
```

Run the xxx.py to test the EBV model
```
$ python xxx.py 
```
