#!/usr/bin/env python
# coding: utf-8

# - Interact with Jupyter Lab to test out ideas
#   - Either locally (maybe sample of the dataset), or on a interactive (CPU) node on HPC to work from /nobackup
#   - Create code that automatically uses the available cores (Ray) or GPU over CPU if available (TensorFlow, Jax)
#   - When ready to train, convert to .py (executable script) and move to HPC for static job

# In[1]:


import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print('Using GPU')
else:
    print('Cannot find a GPU')

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

def preprocess_data(data):
    data_reshaped = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    data_reshaped_normalised = data_reshaped.astype("float32") / 255
    return data_reshaped_normalised

train_images = preprocess_data(train_images)
test_images = preprocess_data(test_images)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_images, train_labels, epochs=1, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")


# In[ ]:





# In[ ]:


module load cuda cudnn
conda activate simple_ml


# In[ ]:




