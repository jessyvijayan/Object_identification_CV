#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os 
from bs4 import BeautifulSoup
from tensorflow.keras.applications.vgg19 import preprocess_input 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,MaxPooling2D,Flatten,Conv2D
import cv2


# In[3]:


labels_path = './annotations/'
images_path = './images/'


# In[4]:


images = sorted(os.listdir(images_path))
labels = sorted(os.listdir(labels_path))


# In[5]:


len(images)


# In[6]:


len(labels)


# In[7]:


labels[500]


# In[8]:


images[500]


# ## Data extraction 

# In[9]:


def generate_images_and_labels(xml_location):
    
    with open(xml_location) as xml_file:
        soup = BeautifulSoup(xml_file.read(),'xml')
        objects = soup.find_all('object')
        no_of_persons = len(objects)
        
        
    # labelling people wearing a mask with 0, not wearing a mask with one and mask worn incorrectly as 2
    labels = []
    boxes = []
    for obj in objects:
        if obj.find('name').text == 'without_mask':
            labels.append(1)
        if obj.find('name').text == 'with_mask':
            labels.append(0)
        if obj.find('name').text == 'mask_weared_incorrect':
            labels.append(2)
           
    # Extracting the position of the image
        x_min = int(obj.find('xmin').text)
        y_min = int(obj.find('ymin').text)
        x_max = int(obj.find('xmax').text)
        y_max = int(obj.find('ymax').text)
        
        boxes.append([x_min,y_min,x_max,y_max])
        
    boxes = np.array(boxes)
    labels = np.array(labels)
    
    # getting the output in form of dictioanry
    result = {}
    result['labels'] = labels
    result['boxes'] = boxes
    
    return result,no_of_persons


# In[10]:


final_result = []
total_no_of_persons = []
for file in labels:
    result,no_of_persons = generate_images_and_labels(labels_path + file)
    print(result)
    print(no_of_persons)
    final_result.append(result)
    total_no_of_persons.append(no_of_persons)
    


# In[11]:


from IPython.display import Image


# In[12]:


Image(filename=images_path+'maksssksksss55.png')


# In[13]:


total_no_of_persons[500]


# In[14]:


final_result[500]


# ## Preprocessing the images

# In[15]:


image_size = 224
face_images = []
face_labels = []

for i,image_path in enumerate(images):
    # reading the image
    image_read = cv2.imread(images_path+image_path)
    # looping every image to find the face location ie xmin,ymin,xmax,ymax and labels 0,1,2
    for j in range(0,total_no_of_persons[i]):
        #takes the location of individual image
        face_loc=final_result[i]['boxes'][j]
        # extracts the xmin,ymin,xmax,ymax coordinates
        temp_face = image_read[face_loc[1]:face_loc[3],face_loc[0]:face_loc[2]]
        # standardising the image size to 224
        temp_face = cv2.resize(temp_face,(image_size,image_size))
        # preprocessing the image as per vgg19 stds
        temp_face = preprocess_input(temp_face)
        # appending preprocessed image to face images
        face_images.append(temp_face)
        # extracting the label information and appending to face labels list
        face_labels.append(final_result[i]['labels'][j])
        
# converting the data into array        
face_images = np.array(face_images)  
face_labels = np.array(face_labels)


# In[16]:


plt.imshow(face_images[0])


# In[17]:


face_labels[22]


# ## Train test split

# In[18]:


x_train,x_test,y_train,y_test = train_test_split(face_images,face_labels,test_size=0.2,shuffle=True,random_state=10)


# In[19]:


image_gen = ImageDataGenerator(zoom_range=0.1,width_shift_range=0.1,height_shift_range=0.1) # done to prevent overfitting


# ## Model Building

# In[20]:


model = Sequential([
    #2 layers of 64 filters each of 3*3 kernel size and 1 maxpooling of 2*2
    Conv2D(64,(3,3), activation='ReLU',padding='same',input_shape = (224,224,3)),
    Conv2D(64,(3,3), activation='ReLU',padding='same'),
    MaxPooling2D(2,2),
    
    #2 layers of 128 filters each of 3*3 kernel size and 1 maxpooling of 2*2
    Conv2D(128,(3,3), activation='ReLU',padding='same'),
    Conv2D(128,(3,3), activation='ReLU',padding='same'),
    MaxPooling2D(2,2),
    
    #3 layers of 256 filters each of 3*3 kernel size and 1 maxpooling of 2*2
    Conv2D(256,(3,3), activation='ReLU',padding='same'),
    Conv2D(256,(3,3), activation='ReLU',padding='same'),
    Conv2D(256,(3,3), activation='ReLU',padding='same'),
    MaxPooling2D(2,2),
    
    #3 layers of 512 filters each of 3*3 kernel size and 1 maxpooling of 2*2
    Conv2D(512,(3,3), activation='ReLU',padding='same'),
    Conv2D(512,(3,3), activation='ReLU',padding='same'),
    Conv2D(512,(3,3), activation='ReLU',padding='same'),
    MaxPooling2D(2,2),   
    
    #3 layers of 512 filters each of 3*3 kernel size and 1 maxpooling of 2*2
    Conv2D(512,(3,3), activation='ReLU',padding='same'),
    Conv2D(512,(3,3), activation='ReLU',padding='same'),
    Conv2D(512,(3,3), activation='ReLU',padding='same'),
    MaxPooling2D(2,2),
    
    #Flattening the layers
    Flatten(),
    
    Dense(2048,activation='relu'),
    Dense(1024,activation='relu'),
    
    # output
    Dense(3,activation='softmax')
])


# In[21]:


model.summary()


# In[22]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[23]:


history = model.fit(image_gen.flow(x_train,y_train),batch_size=200,epochs=3,
                    validation_data=(x_test,y_test))


# In[24]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('accuracy plot')
plt.legend(['training','testing'])


# In[26]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('accuracy plot')
plt.legend(['training','testing'])


# In[ ]:




