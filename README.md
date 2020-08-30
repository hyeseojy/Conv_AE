## Convolutional Autoencoder Implementation
This repository is about Conv AutoEncoder in Tensorflow 2 , I used tf.keras.Model and tf.layers.Layer instead of tf.keras.models.Sequential.  This allows us to customize and have full control of the model, I also used custom training instead of relying on the fit() function.  
In case we have very huge dataset, I applied online loading (by batch) instead of loading the data completely at the beginning. This will eventually not consume the memory.  

#### The Architecrure of Convolutional Autoencoder
<p></p>
<center>
<img src="img/c1.jpg" align="center" width="700" height="400"/>
</center>  

Figure 1: image is taken from [source](https://github.com/arthurmeyer/Saliency_Detection_Convolutional_Autoencoder)

<center>   
<img src="img/c2.png" width="700" height="400"/>   
</center>   

Figure 2: image is taken from [source](https://github.com/ALPHAYA-Japan/autoencoder/blob/master/README.md)

### Training on MNIST
<p></p>
<center>
<img src="img/mnist.png" width="400" height="350"/>
</center>

### Requirement
```
python==3.7.0
numpy==1.18.1
```
### How to use
Training & Prediction can be run as follows:    
`python train.py train`  
`python train.py predict img.png`  


### More information
* Please refer to the original paper of Convolutional AutoEncoder [here](https://web.stanford.edu/class/psych209a/ReadingsByDate/02_06/PDPVolIChapter8.pdf) and [here](https://www.aaai.org/Papers/AAAI/1987/AAAI87-050.pdf) for more information.

### Implementation Notes
* **Note 1**:   
Since datasets are somehow huge and painfully slow in training ,I decided to make number of units variable. If you want to run it in your PC, you can reduce or increase the number of units into any number you like. (512 is by default). For example:  
`model = conv_ae.Conv_AE((None,height, width, channel), latent = 200, units=16)`

* **Note 2** :   
You can also make the size of images smaller, so that it can be ran faster and doesn't take too much memories.

### Result for MNIST:   
* Learning rate = 0.0001
* Batch size = 16  
* Optimizer = Adam   
* units = 16
* latent = 200

Epoch | Training Loss |  Validation Loss  |
:---: | :---: | :---:
1 | 0.0144 | 0.0052
10 | 0.0019 | 0.0017
20 | 0.0008| 0.0008

Epoch | True image and predicted image
:---: | :---:
1 | <img src="img/conv_1.png" />
10 | <img src="img/conv_10.png" />
20 |<img src="img/conv_20.png" />

### Epoch 10
latent | True image and predicted image
:---: | :---:
10 | <img src="img/conv_l_10.png" />
100 | <img src="img/cov_l_100.png" />
200 | <img src="img/conv_10.png" />
300 | <img src="img/conv_l_300.png" />
