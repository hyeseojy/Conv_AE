'''
  Author       : Bao Jiarong
  Creation Date: 2020-08-12
  email        : bao.salirong@gmail.com
  Task         : Convolutional Autoencoder Implementation
  Dataset      : MNIST Digits (0,1,...,9)
'''
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import random
import cv2
import loader
import conv_ae

np.random.seed(7)
tf.random.set_seed(7)
# np.set_printoptions(threshold=np.inf)

# Input/Ouptut Parameters
width      = 24
height     = 24
channel    = 3
model_name = "models/conv_ae/digists"
data_path  = "../../data_img/MNIST/train/"

# Step 0: Global Parameters
epochs     = 10
lr_rate    = 0.0001
batch_size = 16

# Step 1: Create Model
model = conv_ae.Conv_AE((None,height, width, channel), latent = 200, units=16)

if sys.argv[1] == "train":

    print(model.summary())
    # sys.exit()

    # Load weights:
    # model.load_weights(model_name)

    # Step 3: Load data
    X_train, Y_train, X_valid, Y_valid = loader.load_light(data_path,width,height,True,0.8,False)
    # Define The Optimizer
    optimizer= tf.keras.optimizers.Adam(learning_rate=lr_rate)
    # Define The Loss
    #---------------------
    @tf.function
    def my_loss(y_true, y_pred):
        return tf.keras.losses.MSE(y_true=y_true, y_pred=y_pred)

    # Define The Metrics
    tr_loss = tf.keras.metrics.MeanSquaredError(name = 'tr_loss')
    va_loss = tf.keras.metrics.MeanSquaredError(name = 'va_loss')

    #---------------------
    @tf.function
    def train_step(X, Y_true):
        with tf.GradientTape() as tape:
            Y_pred = model(X, training=True)
            loss   = my_loss(y_true=Y_true, y_pred=Y_pred )
        gradients= tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        tr_loss.update_state(y_true = Y_true, y_pred = Y_pred )

    #---------------------
    @tf.function
    def valid_step(X, Y_true):
        Y_pred= model(X, training=False)
        loss  = my_loss(y_true=Y_true, y_pred=Y_pred)
        va_loss.update_state(y_true = Y_true, y_pred = Y_pred)

    #---------------------
    # start training
    L = len(X_train)
    M = len(X_valid)
    steps  = int(L/batch_size)
    steps1 = int(M/batch_size)

    for epoch in range(epochs):
        # Run on training data + Update weights
        for step in range(steps):
            images, labels = loader.get_batch_light(X_train, X_train, batch_size, width, height)
            train_step(images,images)

            print(epoch,"/",epochs,step,steps,
                  "loss:",tr_loss.result().numpy(),end="\r")

        # Run on validation data without updating weights
        for step in range(steps1):
            images, labels = loader.get_batch_light(X_valid, X_valid, batch_size, width, height)
            valid_step(images, images)

        print(epoch,"/",epochs,step,steps,
              "loss:",tr_loss.result().numpy(),
              "val_loss:",va_loss.result().numpy())

        # Save the model for each epoch
        model.save_weights(filepath=model_name, save_format='tf')

elif sys.argv[1] == "predict":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Prepare the input
    img = cv2.imread(sys.argv[2])
    image = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
    images = np.array([image])
    images = loader.scaling_tech(images,method="normalization")

    # Step 5: Predict the class
    preds = my_model.predict(images)
    preds = (preds[0] - preds.min())/(preds.max() - preds.min())
    images = np.hstack((images[0],preds))
    images = cv2.resize(images,(width*4,height*2))
    cv2.imshow("imgs",images)
    cv2.waitKey(0)

elif sys.argv[1] == "predict_all":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Prepare the input
    imgs_filenames = ["../../data_img/MNIST/test/img_2.jpg" , # 0
                      "../../data_img/MNIST/test/img_18.jpg", # 1
                      "../../data_img/MNIST/test/img_1.jpg" , # 2
                      "../../data_img/MNIST/test/img_5.jpg" , # 3
                      "../../data_img/MNIST/test/img_13.jpg", # 4
                      "../../data_img/MNIST/test/img_11.jpg", # 5
                      "../../data_img/MNIST/test/img_35.jpg", # 6
                      "../../data_img/MNIST/test/img_6.jpg" , # 7
                      "../../data_img/MNIST/test/img_45.jpg", # 8
                      "../../data_img/MNIST/test/img_3.jpg" ] # 9
    images = []
    for filename in imgs_filenames:
        img = cv2.imread(filename)
        image = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
        images.append(image)

    # True images
    images = np.array(images)
    images = loader.scaling_tech(images,method="normalization")

    # Predicted images
    preds = my_model.predict(images)
    preds = (preds - preds.min())/(preds.max() - preds.min())


    true_images = np.hstack(images)
    pred_images = np.hstack(preds)

    images = np.vstack((true_images, pred_images))
    h = images.shape[0]
    w = images.shape[1]
    images = cv2.resize(images,(w << 1, h << 1))

    cv2.imshow("imgs",images)
    cv2.waitKey(0)
