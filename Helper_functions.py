from tensorflow.keras.preprocessing.image import ImageDataGenerator

def Augmented_train_data(rotation_range= 0.2,shear_range= 0.2,zoom_range= 0.2,width_shift_range=0.2,height_shift_range= 0.3,horizontal_flip=True):

  train_datagen_augmented = ImageDataGenerator(rescale=1 / 255,
                                               rotation_range, # How much you want to rotate an image ? 
                                               shear_range, # how much do you want to shear in image ? 
                                               zoom_range,# Zoom in image 
                                               width_shift_range, # Move your image around on the x-axis ?
                                               height_shift_range, # Move your image around on the y-axis ?
                                               horizontal_flip # Do you want to flip the image 
                                             )
  return train_datagen_augmented

#######################################################
import tensorflow as tf
import datetime
def create_tensorboard_callback(dir_name, expermint_name):
  log_dir= dir_name+"/"+expermint_name+"/"+ datetime.datetime.now().strftime("Y%m%d%-%H%M%S%")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= log_dir)
  print(f"Saving TensorBoard Log files to : {log_dir}")
  return tensorboard_callback

#######################################################

def count_layers_in_Model(Model):
  for layer_number,layer in enumerate(Model.layers):
    print(layer_number,layer.name)
    
#######################################################
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import datetime


def create_Multiclass_model(Model_URL , Number_Classes ,Train_data,Test_data,Expermint_name,IMAGE_SHAPE):
  
  """
  Model_URL (str) : A tensorflwo Hub feature extraction URL 
  Number_classes (int ): Number of output neurons in the output  layer .
   should be equal to number of target clases . 
  train and test datas.
  expermint_name(str) : type of expermint
  """
  # Create tensorbord callback (functionised coz we need to create a new one for each model)
  def create_tensorboard_callback(dir_name, expermint_name):
 
    
    log_dir= dir_name+"/"+expermint_name+"/"+ datetime.datetime.now().strftime("Y%m%d%-%H%M%S%")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= log_dir)
    #print(f"Saving TensorBoard Log files to : {log_dir}")
    return tensorboard_callback  


  
  
  # Downlod the  Mdoel
  feature_extractor_layer = hub.KerasLayer(Model_URL,
                                           trainable= False,
                                           name="feature_extractor_layer" ,
                                           input_shape= IMAGE_SHAPE+(3,) # this will add 3 to the data shape (224,224,3)


                                            ) # Freeze the already learened Mdoel 

  # Create Mdoel 
  Model = tf.keras.Sequential([feature_extractor_layer,
                               layers.Dense(Number_Classes,activation="softmax",name="output_Layer")])
  # Compile Our MDoel 
  loss_function = tf.keras.losses.CategoricalCrossentropy()
  Model.compile(loss= loss_function,
                     optimizer= tf.keras.optimizers.Adam(),
                     metrics= ["accuracy"])
  History_Model=Model.fit(Train_data,
                 epochs= 5,
                 steps_per_epoch=len(Train_data),
                 validation_data=Test_data,
                 validation_steps=len(Test_data),
                 callbacks= [create_tensorboard_callback(dir_name="tensorflow_hub",
                                                         expermint_name=Expermint_name),
                             
                             ])
  return Model ,History_Model
#######################################################
import zipfile
def unzip_file(filename):

  # unzip file

  zip_ref = zipfile.ZipFile(filename)
  zip_ref.extractall()
  zip_ref.close()
#######################################################
import matplotlib.pyplot as plt

#  plot curve for validation and traning data 
def plot_loss_curves(history):
  """
  Returns seperate loss curves for traning and validation Metrics .
  """
  loss = history.history["loss"]
  val_loss= history.history["val_loss"]
  accuracy= history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  epochs = range(len(history.history["loss"])) # How Many epochs did we run For ?

  # Plot loss 
  plt.plot(epochs,loss,label="Traning_Loss")
  plt.plot(epochs,val_loss,label="val_loss")
  plt.title("epochs")
  plt.legend()
  

  # plot Accuracy 
  plt.figure()
  plt.plot(epochs,accuracy,label= "traning_accuracy")
  plt.plot(epochs,val_accuracy,label= "val_accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend()
  #######################################################
# How many Images in each Folder ?
import os 
def walk_through_data_in_directory(filename):
  # walk through 10 per data directory and list number of files 
  for dirpath,dirnames,filenames  in os.walk(filename):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")