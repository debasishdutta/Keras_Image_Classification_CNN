#!/usr/bin/env python
# coding: utf-8

# In[1]:


### USER INPUT SECTION ###

global_training_dir = 'C:/Users/debas/Downloads/CNN/Train Samples'
global_testing_dir = 'C:/Users/debas/Downloads/CNN/Test Samples'
global_live_testing_dir = 'C:/Users/debas/Downloads/CNN/Live Testing'
global_n_image_train = 8000
global_n_image_test = 2000

param_n_feature_map = 64
param_input_height_pixel = 128 
param_input_weight_pixel = 128
param_input_rgb_channel = 3
param_n_nodes_ann = 256
param_drop_out = 0.1
param_epochs = 10
param_batch_size = 32


# In[2]:


### IMPORT ALL NECCESSARY PACKAGES ###

import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


# In[3]:


### BUILDING A CNN MODEL ###

def model_cnn(training_img_dir,
              testing_img_dir,
              n_image_train,
              n_image_test,
              n_feature_map, 
              input_height_pixel, 
              input_weight_pixel, 
              input_rgb_channel, 
              n_nodes_ann, 
              drop_out, 
              epochs, 
              batch_size):
    
    # Initializing CNN Architecture
    classifier = Sequential()
    
    # Adding First Convolutional Layer
    classifier.add(Conv2D(n_feature_map, (3, 3), 
                          input_shape = (input_height_pixel,input_weight_pixel,input_rgb_channel), 
                          activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding Second Convolutional Layer
    classifier.add(Conv2D(n_feature_map, (3, 3), 
                          activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Flattening Convolutional Layers
    classifier.add(Flatten())
    
    # Fully Connected ANN
    classifier.add(Dense(units = n_nodes_ann, kernel_initializer='uniform', activation = 'relu', bias_initializer='zeros'))
    classifier.add(Dropout(rate = drop_out))
    
    # Output Layer
    classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid', bias_initializer='zeros'))
    
    # Compiling The CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Image Augmentation
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    
    training_set = train_datagen.flow_from_directory(training_img_dir,
                                                     target_size = (input_height_pixel,input_weight_pixel),
                                                     batch_size = batch_size,
                                                     class_mode = 'binary')
    test_set = test_datagen.flow_from_directory(testing_img_dir,
                                                target_size = (input_height_pixel,input_weight_pixel),
                                                batch_size = batch_size,
                                                class_mode = 'binary')
    
    # Running The Model
    classifier.fit_generator(training_set,
                             steps_per_epoch = n_image_train,
                             epochs = epochs,
                             validation_data = test_set,
                             validation_steps = n_image_test)
    
    # Storing Class Encoders
    class_identifier = training_set.class_indices
    class_identifier_df = pd.DataFrame(class_identifier.items(),columns=['Class', 'Encoded'])
    pos_class_name = class_identifier_df.loc[class_identifier_df['Encoded']==1,'Class'].iloc[0]
    neg_class_name = class_identifier_df.loc[class_identifier_df['Encoded']==0,'Class'].iloc[0]
    
    final_list = (classifier,pos_class_name, neg_class_name)
    
    return(final_list)


# In[4]:


### SCORING A CNN MODEL ON NEW DATA ###

def scoring_cnn(live_testing_dir,
                input_height_pixel,
                input_weight_pixel,
                classifier,
                pos_class,
                neg_class):
    
    final_result = []
    for filename in os.listdir(live_testing_dir):
        test_source = live_testing_dir+"/"+filename
        test_image = image.load_img(test_source, target_size = (input_height_pixel,input_weight_pixel))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        test_result = classifier.predict(test_image)[0][0]
        final_result.append(test_result)
    prediction = [pos_class if x == 1 else neg_class for x in final_result]
    print("\nPlease Find The Class Output of Live Images")
    print(prediction)
    
    return(prediction)


# In[ ]:


### SCRIPT EXECUTION ###

model_object = model_cnn(global_training_dir,
                         global_testing_dir,
                         global_n_image_train,
                         global_n_image_test,
                         param_n_feature_map,
                         param_input_height_pixel, 
                         param_input_weight_pixel,
                         param_input_rgb_channel,
                         param_n_nodes_ann,
                         param_drop_out,
                         param_epochs,
                         param_batch_size)

prediction = scoring_cnn(global_live_testing_dir,
                         param_input_height_pixel, 
                         param_input_weight_pixel,
                         model_object[0],
                         model_object[1],
                         model_object[2])


# In[ ]:




