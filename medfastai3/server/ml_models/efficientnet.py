#!/usr/bin/env python
# coding: utf-8

# 

# 

# <a id="import"></a>
# # <center>Import needed modules</center>

# In[1]:


get_ipython().system('pip install -q autoviz')
get_ipython().system('pip install -q -U --pre pycaret')


# In[2]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd
import shutil
import time
import cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from IPython.core.display import display, HTML
# stop annoying tensorflow warning messages
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# <a id="functions"></a>
# # <center>Define Support Functions</center>

# ### Define a function to show example training images

# In[3]:


def show_image_samples(gen ):
    t_dict=gen.class_indices
    classes=list(t_dict.keys())    
    images,labels=next(gen) # get a sample batch from the generator 
    plt.figure(figsize=(20, 20))
    length=len(labels)
    if length<25:   #show maximum of 25 images
        r=length
    else:
        r=25
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image=images[i]/255
        plt.imshow(image)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    plt.show()


# In[4]:


def show_images(tdir):
    classlist=os.listdir(tdir)
    length=len(classlist)
    columns=5
    rows=int(np.ceil(length/columns))    
    plt.figure(figsize=(20, rows * 4))
    for i, klass in enumerate(classlist):    
        classpath=os.path.join(tdir, klass)
        imgpath=os.path.join(classpath, '1.jpg')
        img=plt.imread(imgpath)
        plt.subplot(rows, columns, i+1)
        plt.axis('off')
        plt.title(klass, color='blue', fontsize=12)
        plt.imshow(img)
    


# ### Define  a function to print text in RGB foreground and background colors

# In[5]:


def print_in_color(txt_msg,fore_tupple,back_tupple,):
    #prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple 
    #text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)
    rf,gf,bf=fore_tupple
    rb,gb,bb=back_tupple
    msg='{0}' + txt_msg
    mat='\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m' 
    print(msg .format(mat), flush=True)
    print('\33[0m', flush=True) # returns default print color to back to black
    return


# In[6]:


def F1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# ### Define a subclass of Keras callbacks that will control the learning rate and print
# ### training data in spreadsheet format. The callback also includes a feature to
# ### periodically ask if you want to train for N more epochs or halt

# In[7]:


class LRA(keras.callbacks.Callback):
    def __init__(self,model, base_model, patience,stop_patience, threshold, factor, dwell, batches, initial_epoch,epochs, ask_epoch):
        super(LRA, self).__init__()
        self.model=model
        self.base_model=base_model
        self.patience=patience # specifies how many epochs without improvement before learning rate is adjusted
        self.stop_patience=stop_patience # specifies how many times to adjust lr without improvement to stop training
        self.threshold=threshold # specifies training accuracy threshold when lr will be adjusted based on validation loss
        self.factor=factor # factor by which to reduce the learning rate
        self.dwell=dwell
        self.batches=batches # number of training batch to runn per epoch
        self.initial_epoch=initial_epoch
        self.epochs=epochs
        self.ask_epoch=ask_epoch
        self.ask_epoch_initial=ask_epoch # save this value to restore if restarting training
        # callback variables 
        self.count=0 # how many times lr has been reduced without improvement
        self.stop_count=0        
        self.best_epoch=1   # epoch with the lowest loss        
        self.initial_lr=float(tf.keras.backend.get_value(model.optimizer.lr)) # get the initiallearning rate and save it         
        self.highest_tracc=0.0 # set highest training accuracy to 0 initially
        self.lowest_vloss=np.inf # set lowest validation loss to infinity initially
        self.best_weights=self.model.get_weights() # set best weights to model's initial weights
        self.initial_weights=self.model.get_weights()   # save initial weights if they have to get restored 
        
    def on_train_begin(self, logs=None):        
        if self.base_model != None:
            status=base_model.trainable
            if status:
                msg=' initializing callback starting train with base_model trainable'
            else:
                msg='initializing callback starting training with base_model not trainable'
        else:
            msg='initialing callback and starting training'                        
        print_in_color (msg, (244, 252, 3), (55,65,80)) 
        msg='{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format('Epoch', 'Loss', 'Accuracy',
                                                                                              'V_loss','V_acc', 'LR', 'Next LR', 'Monitor','% Improv', 'Duration')
        print_in_color(msg, (244,252,3), (55,65,80)) 
        self.start_time= time.time()
        
    def on_train_end(self, logs=None):
        stop_time=time.time()
        tr_duration= stop_time- self.start_time            
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))

        self.model.set_weights(self.best_weights) # set the weights of the model to the best weights
        msg=f'Training is completed - model is set with weights from epoch {self.best_epoch} '
        print_in_color(msg, (0,255,0), (55,65,80))
        msg = f'training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
        print_in_color(msg, (0,255,0), (55,65,80))   
        
    def on_train_batch_end(self, batch, logs=None):
        acc=logs.get('accuracy')* 100  # get training accuracy 
        loss=logs.get('loss')
        msg='{0:20s}processing batch {1:4s} of {2:5s} accuracy= {3:8.3f}  loss: {4:8.5f}'.format(' ', str(batch), str(self.batches), acc, loss)
        print(msg, '\r', end='') # prints over on the same line to show running batch count        
        
    def on_epoch_begin(self,epoch, logs=None):
        self.now= time.time()
        
    def on_epoch_end(self, epoch, logs=None):  # method runs on the end of each epoch
        later=time.time()
        duration=later-self.now 
        lr=float(tf.keras.backend.get_value(self.model.optimizer.lr)) # get the current learning rate
        current_lr=lr
        v_loss=logs.get('val_loss')  # get the validation loss for this epoch
        acc=logs.get('accuracy')  # get training accuracy 
        v_acc=logs.get('val_accuracy')
        loss=logs.get('loss')        
        if acc < self.threshold: # if training accuracy is below threshold adjust lr based on training accuracy
            monitor='accuracy'
            if epoch ==0:
                pimprov=0.0
            else:
                pimprov= (acc-self.highest_tracc )*100/self.highest_tracc
            if acc>self.highest_tracc: # training accuracy improved in the epoch                
                self.highest_tracc=acc # set new highest training accuracy
                self.best_weights=self.model.get_weights() # traing accuracy improved so save the weights
                self.count=0 # set count to 0 since training accuracy improved
                self.stop_count=0 # set stop counter to 0
                if v_loss<self.lowest_vloss:
                    self.lowest_vloss=v_loss
                color= (0,255,0)
                self.best_epoch=epoch + 1  # set the value of best epoch for this epoch              
            else: 
                # training accuracy did not improve check if this has happened for patience number of epochs
                # if so adjust learning rate
                if self.count>=self.patience -1: # lr should be adjusted
                    color=(245, 170, 66)
                    lr= lr* self.factor # adjust the learning by factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr) # set the learning rate in the optimizer
                    self.count=0 # reset the count to 0
                    self.stop_count=self.stop_count + 1 # count the number of consecutive lr adjustments
                    self.count=0 # reset counter
                    if self.dwell:
                        self.model.set_weights(self.best_weights) # return to better point in N space                        
                    else:
                        if v_loss<self.lowest_vloss:
                            self.lowest_vloss=v_loss                                    
                else:
                    self.count=self.count +1 # increment patience counter                    
        else: # training accuracy is above threshold so adjust learning rate based on validation loss
            monitor='val_loss'
            if epoch ==0:
                pimprov=0.0
            else:
                pimprov= (self.lowest_vloss- v_loss )*100/self.lowest_vloss
            if v_loss< self.lowest_vloss: # check if the validation loss improved 
                self.lowest_vloss=v_loss # replace lowest validation loss with new validation loss                
                self.best_weights=self.model.get_weights() # validation loss improved so save the weights
                self.count=0 # reset count since validation loss improved  
                self.stop_count=0  
                color=(0,255,0)                
                self.best_epoch=epoch + 1 # set the value of the best epoch to this epoch
            else: # validation loss did not improve
                if self.count>=self.patience-1: # need to adjust lr
                    color=(245, 170, 66)
                    lr=lr * self.factor # adjust the learning rate                    
                    self.stop_count=self.stop_count + 1 # increment stop counter because lr was adjusted 
                    self.count=0 # reset counter
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr) # set the learning rate in the optimizer
                    if self.dwell:
                        self.model.set_weights(self.best_weights) # return to better point in N space
                else: 
                    self.count =self.count +1 # increment the patience counter                    
                if acc>self.highest_tracc:
                    self.highest_tracc= acc
        msg=f'{str(epoch+1):^3s}/{str(self.epochs):4s} {loss:^9.3f}{acc*100:^9.3f}{v_loss:^9.5f}{v_acc*100:^9.3f}{current_lr:^9.5f}{lr:^9.5f}{monitor:^11s}{pimprov:^10.2f}{duration:^8.2f}'
        print_in_color (msg,color, (55,65,80))
        if self.stop_count> self.stop_patience - 1: # check if learning rate has been adjusted stop_count times with no improvement
            msg=f' training has been halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement'
            print_in_color(msg, (0,255,255), (55,65,80))
            self.model.stop_training = True # stop training
        """
        else: 
            if self.ask_epoch !=None:
                if epoch + 1 >= self.ask_epoch:
                    if base_model.trainable:
                        msg='enter H to halt  or an integer for number of epochs to run then ask again'
                    else:
                        msg='enter H to halt ,F to fine tune model, or an integer for number of epochs to run then ask again'
                    print_in_color(msg, (0,255,255), (55,65,80))
                    ans=input('')
                    if ans=='H' or ans=='h':
                        msg=f'training has been halted at epoch {epoch + 1} due to user input'
                        print_in_color(msg, (0,255,255), (55,65,80))
                        self.model.stop_training = True # stop training
                    elif ans == 'F' or ans=='f':
                        if base_model.trainable:
                            msg='base_model is already set as trainable'
                        else:
                            msg='setting base_model as trainable for fine tuning of model'
                            self.base_model.trainable=True
                        print_in_color(msg, (0, 255,255), (55,65,80))
                        msg='{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:^8s}'.format('Epoch', 'Loss', 'Accuracy',
                                                                                              'V_loss','V_acc', 'LR', 'Next LR', 'Monitor','% Improv', 'Duration')
                        print_in_color(msg, (244,252,3), (55,65,80))                         
                        self.count=0
                        self.stop_count=0                        
                        self.ask_epoch = epoch + 1 + self.ask_epoch_initial 
                        
                    else:
                        ans=int(ans)
                        self.ask_epoch +=ans
                        msg=f' training will continue until epoch ' + str(self.ask_epoch)                         
                        print_in_color(msg, (0, 255,255), (55,65,80))
                        msg='{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format('Epoch', 'Loss', 'Accuracy',
                                                                                              'V_loss','V_acc', 'LR', 'Next LR', 'Monitor','% Improv', 'Duration')
                        print_in_color(msg, (244,252,3), (55,65,80)) 
                        """


# ### Define a function to plot the training data

# In[8]:


def tr_plot(tr_data, start_epoch):
    #Plot the training and validation data
    tacc=tr_data.history['accuracy']
    tloss=tr_data.history['loss']
    vacc=tr_data.history['val_accuracy']
    vloss=tr_data.history['val_loss']
    Epoch_count=len(tacc)+ start_epoch
    Epochs=[]
    for i in range (start_epoch ,Epoch_count):
        Epochs.append(i+1)   
    index_loss=np.argmin(vloss)#  this is the epoch with the lowest validation loss
    val_lowest=vloss[index_loss]
    index_acc=np.argmax(vacc)
    acc_highest=vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label='best epoch= '+ str(index_loss+1 +start_epoch)
    vc_label='best epoch= '+ str(index_acc + 1+ start_epoch)
    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(20,8))
    axes[0].plot(Epochs,tloss, 'r', label='Training loss')
    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )
    axes[0].scatter(index_loss+1 +start_epoch,val_lowest, s=150, c= 'blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
    axes[1].scatter(index_acc+1 +start_epoch,acc_highest, s=150, c= 'blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    #plt.style.use('fivethirtyeight')
    plt.show()


# ### define a function to create confusion matrix and classification report

# In[9]:


def print_in_color(txt_msg,fore_tupple=(0,255,255),back_tupple=(100,100,100)):
    #prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple 
    #text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)
    # default parameter print in cyan foreground and gray background
    rf,gf,bf=fore_tupple
    rb,gb,bb=back_tupple
    msg='{0}' + txt_msg
    mat='\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m' 
    print(msg .format(mat), flush=True)
    print('\33[0m', flush=True) # returns default print color to back to black
    return

# example default print
msg='test of default colors'
print_in_color(msg)


# In[10]:


def print_info( test_gen, preds, print_code, save_dir, subject ):
    class_dict=test_gen.class_indices
    labels= test_gen.labels
    file_names= test_gen.filenames 
    error_list=[]
    true_class=[]
    pred_class=[]
    prob_list=[]
    new_dict={}
    error_indices=[]
    y_pred=[]
    for key,value in class_dict.items():
        new_dict[value]=key             # dictionary {integer of class number: string of class name}
    # store new_dict as a text fine in the save_dir
    classes=list(new_dict.values())     # list of string of class names     
    errors=0      
    for i, p in enumerate(preds):
        pred_index=np.argmax(p)         
        true_index=labels[i]  # labels are integer values
        if pred_index != true_index: # a misclassification has occurred
            error_list.append(file_names[i])
            true_class.append(new_dict[true_index])
            pred_class.append(new_dict[pred_index])
            prob_list.append(p[pred_index])
            error_indices.append(true_index)            
            errors=errors + 1
        y_pred.append(pred_index)    
    if print_code !=0:
        if errors>0:
            if print_code>errors:
                r=errors
            else:
                r=print_code           
            msg='{0:^28s}{1:^28s}{2:^28s}{3:^16s}'.format('Filename', 'Predicted Class' , 'True Class', 'Probability')
            print_in_color(msg, (0,255,0),(55,65,80))
            for i in range(r):                
                split1=os.path.split(error_list[i])                
                split2=os.path.split(split1[0])                
                fname=split2[1] + '/' + split1[1]
                msg='{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(fname, pred_class[i],true_class[i], ' ', prob_list[i])
                print_in_color(msg, (255,255,255), (55,65,60))
                #print(error_list[i]  , pred_class[i], true_class[i], prob_list[i])               
        else:
            msg='With accuracy of 100 % there are no errors to print'
            print_in_color(msg, (0,255,0),(55,65,80))
    if errors>0:
        plot_bar=[]
        plot_class=[]
        for  key, value in new_dict.items():        
            count=error_indices.count(key) 
            if count!=0:
                plot_bar.append(count) # list containg how many times a class c had an error
                plot_class.append(value)   # stores the class 
        fig=plt.figure()
        fig.set_figheight(len(plot_class)/3)
        fig.set_figwidth(10)
        plt.style.use('fivethirtyeight')
        for i in range(0, len(plot_class)):
            c=plot_class[i]
            x=plot_bar[i]
            plt.barh(c, x, )
            plt.title( ' Errors by Class on Test Set')
    y_true= np.array(labels)        
    y_pred=np.array(y_pred)
    if len(classes)<= 30:
        # create a confusion matrix 
        cm = confusion_matrix(y_true, y_pred )        
        length=len(classes)
        if length<8:
            fig_width=8
            fig_height=8
        else:
            fig_width= int(length * .5)
            fig_height= int(length * .5)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)       
        plt.xticks(np.arange(length)+.5, classes, rotation= 90)
        plt.yticks(np.arange(length)+.5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:\n----------------------\n", clr)
    return cm,y_true


# ### define a function to save the model and the associated class_dict.csv file

# In[11]:


def saver(save_path, model, model_name, subject, accuracy,img_size, scalar, generator):
    # first save the model
    save_id=str (model_name +  '-' + subject +'-'+ str(acc)[:str(acc).rfind('.')+3] + '.h5')
    model_save_loc=os.path.join(save_path, save_id)
    model.save(model_save_loc)
    print_in_color ('model was saved as ' + model_save_loc, (0,255,0),(55,65,80)) 
    # now create the class_df and convert to csv file    
    class_dict=generator.class_indices 
    height=[]
    width=[]
    scale=[]
    for i in range(len(class_dict)):
        height.append(img_size[0])
        width.append(img_size[1])
        scale.append(scalar)
    Index_series=pd.Series(list(class_dict.values()), name='class_index')
    Class_series=pd.Series(list(class_dict.keys()), name='class') 
    Height_series=pd.Series(height, name='height')
    Width_series=pd.Series(width, name='width')
    Scale_series=pd.Series(scale, name='scale by')
    class_df=pd.concat([Index_series, Class_series, Height_series, Width_series, Scale_series], axis=1)    
    csv_name='class_dict.csv'
    csv_save_loc=os.path.join(save_path, csv_name)
    class_df.to_csv(csv_save_loc, index=False) 
    print_in_color ('class csv file was saved as ' + csv_save_loc, (0,255,0),(55,65,80)) 
    return model_save_loc, csv_save_loc


# ### define a function that uses the trained model and the
# ### class_dict.csv file to predict images

# In[12]:


def predictor(sdir, csv_path,  model_path, averaged=True, verbose=True):    
    # read in the csv file
    class_df=pd.read_csv(csv_path)    
    class_count=len(class_df['class'].unique())
    img_height=int(class_df['height'].iloc[0])
    img_width =int(class_df['width'].iloc[0])
    img_size=(img_width, img_height)    
    scale=class_df['scale by'].iloc[0] 
    image_list=[]
    # determine value to scale image pixels by
    try: 
        s=int(scale)
        s2=1
        s1=0
    except:
        split=scale.split('-')
        s1=float(split[1])
        s2=float(split[0].split('*')[1])
    path_list=[]
    paths=os.listdir(sdir)    
    for f in paths:
        path_list.append(os.path.join(sdir,f))
    if verbose:
        print (' Model is being loaded- this will take about 10 seconds')
    model=load_model(model_path)
    image_count=len(path_list) 
    image_list=[]
    file_list=[]
    good_image_count=0
    for i in range (image_count):        
        try:
            img=cv2.imread(path_list[i])
            img=cv2.resize(img, img_size)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
            good_image_count +=1
            img=img*s2 - s1             
            image_list.append(img)
            file_name=os.path.split(path_list[i])[1]
            file_list.append(file_name)
        except:
            if verbose:
                print ( path_list[i], ' is an invalid image file')
    if good_image_count==1: # if only a single image need to expand dimensions
        averaged=True
    image_array=np.array(image_list)    
    # make predictions on images, sum the probabilities of each class then find class index with
    # highest probability
    preds=model.predict(image_array)    
    if averaged:
        psum=[]
        for i in range (class_count): # create all 0 values list
            psum.append(0)    
        for p in preds: # iterate over all predictions
            for i in range (class_count):
                psum[i]=psum[i] + p[i]  # sum the probabilities   
        index=np.argmax(psum) # find the class index with the highest probability sum        
        klass=class_df['class'].iloc[index] # get the class name that corresponds to the index
        prob=psum[index]/good_image_count  # get the probability average         
        # to show the correct image run predict again and select first image that has same index
        for img in image_array:  #iterate through the images    
            test_img=np.expand_dims(img, axis=0) # since it is a single image expand dimensions 
            test_index=np.argmax(model.predict(test_img)) # for this image find the class index with highest probability
            if test_index== index: # see if this image has the same index as was selected previously
                if verbose: # show image and print result if verbose=1
                    plt.axis('off')
                    plt.imshow(img) # show the image
                    print (f'predicted species is {klass} with a probability of {prob:6.4f} ')
                break # found an image that represents the predicted class      
        return klass, prob, img, None
    else: # create individual predictions for each image
        pred_class=[]
        prob_list=[]
        for i, p in enumerate(preds):
            index=np.argmax(p) # find the class index with the highest probability sum
            klass=class_df['class'].iloc[index] # get the class name that corresponds to the index
            image_file= file_list[i]
            pred_class.append(klass)
            prob_list.append(p[index])            
        Fseries=pd.Series(file_list, name='image file')
        Lseries=pd.Series(pred_class, name= 'species')
        Pseries=pd.Series(prob_list, name='probability')
        df=pd.concat([Fseries, Lseries, Pseries], axis=1)
        if verbose:
            length= len(df)
            print (df.head(length))
        return None, None, None, df


# ### define a function tha takes in a dataframe df, and integer max_size and a string column
# ### and returns a dataframe where the number of samples for any class specified by column
# ### is limited to max samples

# In[13]:


def trim (df, max_size, min_size, column):
    df=df.copy()
    original_class_count= len(list(df[column].unique()))
    print ('Original Number of classes in dataframe: ', original_class_count)
    sample_list=[] 
    groups=df.groupby(column)
    for label in df[column].unique():        
        group=groups.get_group(label)
        sample_count=len(group)         
        if sample_count> max_size :
            strat=group[column]
            samples,_=train_test_split(group, train_size=max_size, shuffle=True, random_state=123, stratify=strat)            
            sample_list.append(samples)
        elif sample_count>= min_size:
            sample_list.append(group)
    df=pd.concat(sample_list, axis=0).reset_index(drop=True)
    final_class_count= len(list(df[column].unique())) 
    if final_class_count != original_class_count:
        print ('*** WARNING***  dataframe has a reduced number of classes' )
    balance=list(df[column].value_counts())
    print (balance)
    return df


# ### define a function that takes in a dataframe, and integers max_samples, min_samples. 
# it uses the function trim to set the maximum number of samples in a class defined by the string column to max_samples.
# if the number of samples is less than min_samples the class is eliminated from the dataset. If some classes have
# less than max_samples, then augmented images are created for that class  and stored in the working_dir so the class
# will have max_samples of images.  After augmentation an aug_df is created for the augmented images in the
# working_dir. The aug_df is then merged with the original train_df to produce a new train_df that has exactly
# max_sample images in each class thus creating a balanced training set.

# In[14]:


def balance(train_df,max_samples, min_samples, column, working_dir, image_size):
    train_df=train_df.copy()
    train_df=trim (train_df, max_samples, min_samples, column)    
    # make directories to store augmented images
    aug_dir=os.path.join(working_dir, 'aug')
    if os.path.isdir(aug_dir):
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in train_df['labels'].unique():    
        dir_path=os.path.join(aug_dir,label)    
        os.mkdir(dir_path)
    # create and store the augmented images  
    total=0
    gen=ImageDataGenerator(horizontal_flip=True,  rotation_range=20, width_shift_range=.2,
                                  height_shift_range=.2, zoom_range=.2)
    groups=train_df.groupby('labels') # group by class
    for label in train_df['labels'].unique():  # for every class               
        group=groups.get_group(label)  # a dataframe holding only rows with the specified label 
        sample_count=len(group)   # determine how many samples there are in this class  
        if sample_count< max_samples: # if the class has less than target number of images
            aug_img_count=0
            delta=max_samples-sample_count  # number of augmented images to create
            target_dir=os.path.join(aug_dir, label)  # define where to write the images    
            aug_gen=gen.flow_from_dataframe( group,  x_col='filepaths', y_col=None, target_size=image_size,
                                            class_mode=None, batch_size=1, shuffle=False, 
                                            save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                            save_format='jpg')
            while aug_img_count<delta:
                images=next(aug_gen)            
                aug_img_count += len(images)
            total +=aug_img_count
    print('Total Augmented images created= ', total)
    # create aug_df and merge with train_df to create composite training set ndf
    if total>0:
        aug_fpaths=[]
        aug_labels=[]
        classlist=os.listdir(aug_dir)
        for klass in classlist:
            classpath=os.path.join(aug_dir, klass)     
            flist=os.listdir(classpath)    
            for f in flist:        
                fpath=os.path.join(classpath,f)         
                aug_fpaths.append(fpath)
                aug_labels.append(klass)
        Fseries=pd.Series(aug_fpaths, name='filepaths')
        Lseries=pd.Series(aug_labels, name='labels')
        aug_df=pd.concat([Fseries, Lseries], axis=1)
        train_df=pd.concat([train_df,aug_df], axis=0).reset_index(drop=True)
        print (list(train_df['labels'].value_counts()) )
    return train_df 


# <a id="image"></a>
# # <center>Input an MRI Image of No tumor and Get the Shape</center><a id="callback"></a>

# In[15]:


"""
img_path=r'../input/brain-tumor-detection/no/No12.jpg'
img=plt.imread(img_path)
print (img.shape)
plt.axis('off')
plt.imshow(img)
plt.show()
"""


# <a id="data"></a>
# # <center>Iterate Through the Images and Create train, test and valid data frames</center><a id="callback"></a>

# In[16]:


def preprocess (sdir, trsplit, vsplit, random_seed):
    filepaths=[]
    labels=[]    
    classlist=os.listdir(sdir)
    for klass in classlist:
        if klass== 'no' or klass =='yes':
            classpath=os.path.join(sdir,klass)
            flist=os.listdir(classpath)
            for f in flist:
                fpath=os.path.join(classpath,f)
                filepaths.append(fpath)
                labels.append(klass)
    Fseries=pd.Series(filepaths, name='filepaths')
    Lseries=pd.Series(labels, name='labels')
    df=pd.concat([Fseries, Lseries], axis=1)       
    # split df into train_df and test_df 
    dsplit=vsplit/(1-trsplit)
    strat=df['labels']    
    train_df, dummy_df=train_test_split(df, train_size=trsplit, shuffle=True, random_state=random_seed, stratify=strat)
    strat=dummy_df['labels']
    valid_df, test_df=train_test_split(dummy_df, train_size=dsplit, shuffle=True, random_state=random_seed, stratify=strat)
    print('train_df length: ', len(train_df), '  test_df length: ',len(test_df), '  valid_df length: ', len(valid_df))
     # check that each dataframe has the same number of classes to prevent model.fit errors
    trcount=len(train_df['labels'].unique())
    tecount=len(test_df['labels'].unique())
    vcount=len(valid_df['labels'].unique())
    if trcount != tecount :         
        msg='** WARNING ** number of classes in training set not equal to number of classes in test set'
        print_in_color(msg, (255,0,0), (55,65,80))
        msg='This will throw an error in either model.evaluate or model.predict'
        print_in_color(msg, (255,0,0), (55,65,80))
    if trcount != vcount:
        msg='** WARNING ** number of classes in training set not equal to number of classes in validation set' 
        print_in_color(msg, (255,0,0), (55,65,80))
        msg=' this will throw an error in model.fit'
        print_in_color(msg, (255,0,0), (55,65,80))
        print ('train df class count: ', trcount, 'test df class count: ', tecount, ' valid df class count: ', vcount) 
        ans=input('Enter C to continue execution or H to halt execution')
        if ans =='H' or ans == 'h':
            print_in_color('Halting Execution', (255,0,0), (55,65,80))
            import sys
            sys.exit('program halted by user')            
    print(list(train_df['labels'].value_counts()))
    return train_df, test_df, valid_df
    


# In[17]:


def preprocess2 (sdir, trsplit, vsplit, random_seed):
    filepaths=[]
    labels=[]    
    classlist=os.listdir(sdir)
    for klass in classlist:
        if klass== 'nn_tumor_gen' or klass =='tumor_gen':
            classpath=os.path.join(sdir,klass)
            flist=os.listdir(classpath)
            for f in flist:
                fpath=os.path.join(classpath,f)
                filepaths.append(fpath)
                labels.append(klass)
    Fseries=pd.Series(filepaths, name='filepaths')
    Lseries=pd.Series(labels, name='labels')
    df=pd.concat([Fseries, Lseries], axis=1)       
    # split df into train_df and test_df 
    dsplit=vsplit/(1-trsplit)
    strat=df['labels']    
    train_df, dummy_df=train_test_split(df, train_size=trsplit, shuffle=True, random_state=random_seed, stratify=strat)
    strat=dummy_df['labels']
    valid_df, test_df=train_test_split(dummy_df, train_size=dsplit, shuffle=True, random_state=random_seed, stratify=strat)
    print('train_df length: ', len(train_df), '  test_df length: ',len(test_df), '  valid_df length: ', len(valid_df))
     # check that each dataframe has the same number of classes to prevent model.fit errors
    trcount=len(train_df['labels'].unique())
    tecount=len(test_df['labels'].unique())
    vcount=len(valid_df['labels'].unique())
    if trcount != tecount :         
        msg='** WARNING ** number of classes in training set not equal to number of classes in test set'
        print_in_color(msg, (255,0,0), (55,65,80))
        msg='This will throw an error in either model.evaluate or model.predict'
        print_in_color(msg, (255,0,0), (55,65,80))
    if trcount != vcount:
        msg='** WARNING ** number of classes in training set not equal to number of classes in validation set' 
        print_in_color(msg, (255,0,0), (55,65,80))
        msg=' this will throw an error in model.fit'
        print_in_color(msg, (255,0,0), (55,65,80))
        print ('train df class count: ', trcount, 'test df class count: ', tecount, ' valid df class count: ', vcount) 
        ans=input('Enter C to continue execution or H to halt execution')
        if ans =='H' or ans == 'h':
            print_in_color('Halting Execution', (255,0,0), (55,65,80))
            import sys
            sys.exit('program halted by user')            
    print(list(train_df['labels'].value_counts()))
    return train_df, test_df, valid_df
    
    


# In[18]:


#sdir1=r'../input/brain-tumor-detection'
#sdir = '/kaggle/input/brain-mri-images-for-brain-tumor-detection/'

#train_df, test_df, valid_df= preprocess(sdir1, .8,.1, 123)


# In[19]:


#sdir2=r'/kaggle/input/bratstraining/training2classes - Copie'
#sdir = '/kaggle/input/brain-mri-images-for-brain-tumor-detection/'

#train_df2, test_df2, valid_df2= preprocess2(sdir2, .8,.1, 123)


# In[20]:


#testdfpaper=pd.concat([train_df2, test_df2, valid_df2],ignore_index = True)


# In[21]:


#testdfpaper


# ### train_df is  balanced 

# <a id="generators"></a>
# # <center>Create train, test and validation generators</center><a id="callback"></a>

# In[22]:


"""
def scalar(img):    
    return img  # EfficientNet expects pixelsin range 0 to 255 so no scaling is required
trgen=ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
tvgen=ImageDataGenerator(preprocessing_function=scalar)
train_gen=trgen.flow_from_dataframe( train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=True, batch_size=batch_size)
test_gen=tvgen.flow_from_dataframe( test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=False, batch_size=test_batch_size)

valid_gen=tvgen.flow_from_dataframe( valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=True, batch_size=batch_size)
classes=list(train_gen.class_indices.keys())
class_count=len(classes)
train_steps=int(np.ceil(len(train_gen.labels)/batch_size))
"""


# In[23]:


#testdfpaper


# In[24]:


#test_gen_paper=tvgen.flow_from_dataframe( testdfpaper, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
 #                                   color_mode='rgb', shuffle=False, batch_size=test_batch_size)


# <a id="show"></a>
# # <center>Show some training images</center>

# In[ ]:





# In[25]:


def define_paths(data_dir):
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            if fold == '1':
                labels.append('meningioma')
            elif fold == '2':
                labels.append('glioma')
            elif fold == '3':
                labels.append('pituitary tumor')

    return filepaths, labels


# Concatenate data paths with labels into one dataframe ( to later be fitted into the model )
def define_df(files, classes):
    Fseries = pd.Series(files, name= 'filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis= 1)


# Split dataframe to train, valid, and test
def create_df(data_dir):
    # train dataframe
    files, classes = define_paths(data_dir)
    df = define_df(files, classes)
    strat = df['labels']
    train_df, dummy_df = train_test_split(df,  train_size= 0.8, shuffle= True, random_state= 123, stratify= strat)

    # valid and test dataframe
    strat = dummy_df['labels']
    valid_df, test_df = train_test_split(dummy_df,  train_size= 0.5, shuffle= True, random_state= 123, stratify= strat)

    return train_df, valid_df, test_df


# In[ ]:





# In[ ]:





# In[26]:


def create_gens (train_df, valid_df, test_df, batch_size):
    '''
    This function takes train, validation, and test dataframe and fit them into image data generator, because model takes data from image data generator.
    Image data generator converts images into tensors. '''


    # define model parameters
    img_size = (224, 224)
    channels = 3 # either BGR or Grayscale
    color = 'rgb'
    img_shape = (img_size[0], img_size[1], channels)

    # Recommended : use custom function for test data batch size, else we can use normal batch size.
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size

    # This function which will be used in image data generator for data augmentation, it just take the image and return it again.
    def scalar(img):
        return img

    tr_gen = ImageDataGenerator(preprocessing_function= scalar, horizontal_flip= True)
    ts_gen = ImageDataGenerator(preprocessing_function= scalar)

    train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

    valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

    # Note: we will use custom test_batch_size, and make shuffle= false
    test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= False, batch_size= test_batch_size)

    return train_gen, valid_gen, test_gen


# In[27]:


data_dir = '/kaggle/input/brain-tumor'

try:
    # Get splitted data
    train_df3, valid_df3, test_df3 = create_df(data_dir)

    # Get Generators
    batch_size = 40
    train_gen3, valid_gen3, test_gen3 = create_gens(train_df3, valid_df3, test_df3, batch_size)

except:
    print('Invalid Input')


# In[ ]:





# In[28]:


train_df=train_df3
valid_df=valid_df3
test_df= test_df3


# In[29]:


train_gen=train_gen3
valid_gen=valid_gen3
test_gen=test_gen3


# In[30]:


# Create Model Structure
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_gen.class_indices.keys()))


# In[31]:


#show_image_samples(train_gen)


# In[32]:


working_dir = r'./'
img_size=(300,300)
channels=3
batch_size=20
img_shape=(img_size[0], img_size[1], channels)
length=len(test_df)
test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]  
test_steps=int(length/test_batch_size)
print ( 'test batch size: ' ,test_batch_size, '  test steps: ', test_steps)


# In[33]:


batch_size = 40   # set batch size for training
epochs = 40   # number of all epochs in training
patience = 1   #number of epochs to wait to adjust lr if monitored value does not improve
stop_patience = 3   # number of epochs to wait before stopping training if monitored value does not improve
threshold = 0.9   # if train accuracy is < threshold adjust monitor accuracy, else monitor validation loss
factor = 0.5   # factor to reduce lr by
#ask_epoch = 5   # number of epochs to run before asking if you want to halt training
batches = int(np.ceil(len(train_gen.labels) / batch_size)) 


# In[34]:


def plot_training(hist,losspath,accuracypath):
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    plt.savefig(losspath)
    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'
    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout
    plt.savefig(accuracypath)
    plt.show()


# In[35]:


import pandas as pd
import numpy as np
from scipy import interp

from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

def class_report(name,y_true, y_pred,y_score=None, average='micro'):
    """
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return
"""
    lb = LabelBinarizer()

    #if len(y_true.shape) == 1:
    lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum() 
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total
    path= "Report of: "+ name +".csv"

    class_report_df.to_csv(path)

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int), 
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), 
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(), 
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"], 
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)


        class_report_df.to_csv(path)

    return class_report_df


# In[36]:


"""
def plot_loss_accuracy(history,newloss,newaccuray):
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    
    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'fontname':'Arial', 'size':'14'}
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    #epochs = range(len(acc))
    #plt.style.use('fivethirtyeight')
    Epochs = [i+1 for i in range(len(acc))]
    plt.figure(figsize= (15, 15))

    plt.plot(Epochs, history.history["accuracy"], 'r', label= 'Training accuracy')
    plt.plot(Epochs, history.history['val_accuracy'], 'g', label= 'Validation accuracy')
   
    #plt.plot(history.history["accuracy"], 'bo--', label="accuracy")
    #plt.plot(history.history['val_accuracy'], 'ro--', label="validation accuracy")
    plt.title('Training and Validation accuracy',fontsize=20)
    plt.ylabel('accuracy',fontsize=20)
    plt.xlabel('Epochs',fontsize=20)
    plt.legend()

    plt.savefig(newaccuray,dpi=1200)
    #plt.figure()
    plt.show()
    
    #plt.style.use('fivethirtyeight')

    #plt.plot(history.history["loss"], "bo--", label="loss")
    #plt.plot(history.history["val_loss"], "ro--", label = "validation loss")
    plt.figure(figsize= (10, 10))

    plt.plot(Epochs, history.history["loss"], 'r', label= 'Training loss')
    plt.plot(Epochs, history.history["val_loss"], 'g', label= 'Validation loss')
    
    plt.title('Training and Validation loss')
    plt.ylabel('loss',fontsize=20)
    plt.xlabel('Epochs',fontsize=20)
    plt.legend()
    plt.savefig(newloss,dpi=1200)
    plt.show()
"""


# In[37]:


def plot_loss_accuracy(history,newloss, newaccuray):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    

    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(newaccuray)
    #plt.figure()
    plt.show()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(newloss)
    plt.show()


# In[38]:


def plot_loss_accuracy2(history,newloss,newaccuray):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    
    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'fontname':'Arial', 'size':'14'}
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    #epochs = range(len(acc))
    plt.style.use('fivethirtyeight')
    Epochs = [i+1 for i in range(len(acc))]
    #plt.figure(figsize= (15, 15))
    ax.xaxis.label.set_size(20)
    plt.plot(Epochs, history.history["accuracy"], 'r', label= 'Training accuracy')
    plt.plot(Epochs, history.history['val_accuracy'], 'g', label= 'Validation accuracy')
   
    #plt.plot(history.history["accuracy"], 'bo--', label="accuracy")
    #plt.plot(history.history['val_accuracy'], 'ro--', label="validation accuracy")
    ax.set_title('Training and Validation accuracy',fontweight="bold", size=20)
    ax.set_ylabel('accuracy',fontsize=25)
    ax.set_xlabel('Epochs',fontsize=25)
    ax.yaxis.set_tick_params(labelsize='large')
    ax.xaxis.set_tick_params(labelsize='large')
    ax.legend()

    fig.savefig(newaccuray,dpi=1200)
    #plt.figure()
    plt.show()
    
    plt.style.use('fivethirtyeight')

    #plt.plot(history.history["loss"], "bo--", label="loss")
    #plt.plot(history.history["val_loss"], "ro--", label = "validation loss")
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    plt.plot(Epochs, history.history["loss"], 'r', label= 'Training loss')
    plt.plot(Epochs, history.history["val_loss"], 'g', label= 'Validation loss')
    
    ax.set_title('Training and Validation loss')
    ax.set_ylabel('loss',fontsize=20)
    ax.set_xlabel('Epochs',fontsize=20)
    ax.yaxis.set_tick_params(labelsize='large')
    ax.xaxis.set_tick_params(labelsize='large')
    plt.legend()
    fig.savefig(newloss,dpi=1200)
    plt.show()


# In[39]:


import pandas as pd
import numpy as np
from scipy import interp

from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

def class_report(name,y_true, y_pred,y_score=None, average='micro'):
    """
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return
"""
    lb = LabelBinarizer()

    #if len(y_true.shape) == 1:
    lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum() 
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total
    path= "Report of: "+ name +".csv"

    #class_report_df.to_csv(path)

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int), 
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), 
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(), 
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"], 
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)


        #class_report_df.to_csv(path)

    return class_report_df


# In[40]:


def plot_cm3(cm,name):
    classes = ['meningioma', 'glioma','pituitary tumor']
    class_count=3
    plt.figure(figsize=(5,5))
    ax = sns.heatmap(cm, annot = True,vmin=0, xticklabels = classes, yticklabels = classes, annot_kws={'size':20}, fmt='d', cmap='cividis',cbar=False)

    ax.set_title('Confusion matrix')
    
    ax.set_xlabel('\nPredicted Values',fontweight ='bold',fontsize = 14)
    ax.set_ylabel('Actual Values ',fontweight ='bold',fontsize = 14);

    ax.xaxis.set_ticklabels(['meningioma', 'glioma','pituitary tumor'])
    ax.yaxis.set_ticklabels(['meningioma', 'glioma','pituitary tumor'])
    plt.xticks(np.arange(class_count)+.5, classes, rotation=45,fontweight ="bold",fontsize = 12)
    plt.yticks(np.arange(class_count)+.5, classes, rotation=0,fontweight ="bold",fontsize = 12)
    path=  "CM"+name+".png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0.5)
    plt.show()


# In[41]:


def plot_cm(cm,name):
    classes = ['No_tumer', 'Tumer']
    class_count=2
    plt.figure(figsize=(5,5))
    ax = sns.heatmap(cm, annot = True,vmin=0, xticklabels = classes, yticklabels = classes, annot_kws={'size':20}, fmt='d', cmap='Purples',cbar=False)

    ax.set_title('Confusion matrix')
    
    ax.set_xlabel('\nPredicted Values',fontweight ='bold',fontsize = 14)
    ax.set_ylabel('Actual Values ',fontweight ='bold',fontsize = 14);

    ax.xaxis.set_ticklabels(['No_tumer', 'Tumer'])
    ax.yaxis.set_ticklabels(['No_tumer', 'Tumer'])
    plt.xticks(np.arange(class_count)+.5, classes, rotation=45,fontweight ="bold",fontsize = 12)
    plt.yticks(np.arange(class_count)+.5, classes, rotation=0,fontweight ="bold",fontsize = 12)
    path=  "CM"+name+".png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0.5)
    plt.show()


# In[42]:


def plot_cm_Dynamic3(cm,name):
    classes = ['meningioma', 'glioma','pituitary tumor']
    class_count=3
    plt.figure(figsize=(5,5))
    ax = sns.heatmap(cm, annot = True,vmin=0, xticklabels = classes, yticklabels = classes, annot_kws={'size':20}, fmt='d', cmap='BuPu',cbar=False)

    ax.set_title('Confusion matrix')
    
    ax.set_xlabel('\nPredicted Values',fontweight ='bold',fontsize = 14)
    ax.set_ylabel('Actual Values ',fontweight ='bold',fontsize = 14);

    ax.xaxis.set_ticklabels(['meningioma', 'glioma','pituitary tumor'])
    ax.yaxis.set_ticklabels(['meningioma', 'glioma','pituitary tumor'])
    plt.xticks(np.arange(class_count)+.5, classes, rotation=45,fontweight ="bold",fontsize = 12)
    plt.yticks(np.arange(class_count)+.5, classes, rotation=0,fontweight ="bold",fontsize = 12)
    path=  "CM"+name+".png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0.5)
    plt.show()


# In[43]:


def plot_cm_Dynamic(cm,name):
    classes = ['No_tumer', 'Tumer']
    class_count=2
    plt.figure(figsize=(5,5))
    ax = sns.heatmap(cm, annot = True,vmin=0, xticklabels = classes, yticklabels = classes, annot_kws={'size':20}, fmt='d', cmap='Purples',cbar=False)

    ax.set_title('Confusion matrix')
    
    ax.set_xlabel('\nPredicted Values',fontweight ='bold',fontsize = 14)
    ax.set_ylabel('Actual Values ',fontweight ='bold',fontsize = 14);

    ax.xaxis.set_ticklabels(['No_tumer', 'Tumer'])
    ax.yaxis.set_ticklabels(['No_tumer', 'Tumer'])
    plt.xticks(np.arange(class_count)+.5, classes, rotation=45,fontweight ="bold",fontsize = 12)
    plt.yticks(np.arange(class_count)+.5, classes, rotation=0,fontweight ="bold",fontsize = 12)
    path=  "CM"+name+".png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0.5)
    plt.show()


# In[44]:


from sklearn.metrics import confusion_matrix

def make_confusion_matrix(cf,name,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):

    # Check if the input is a DataFrame
    if isinstance(cf, pd.DataFrame):
        cf = cf.values

    # Add 'Total Predicted' column and 'Total Actual' row to the matrix
    total_predicted = np.sum(cf, axis=0)
    total_actual = np.sum(cf, axis=1)
    
    cf = np.vstack([cf, total_predicted])
    cf = np.hstack([cf, np.append(total_actual, np.sum(total_actual)).reshape(-1, 1)])
    grand_total = cf[-1, -1]

    # Set the last row and column to custom colormap
    c = cf.copy()
    c[:-1, :-1] = 0
    masked_cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)

    # Adjust box labels
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / grand_total]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    
    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:
        categories = False
    
    if categories == 'auto':
        x_labels = list(range(cf.shape[1] - 1))  # excluding the 'Total' column
        y_labels = list(range(cf.shape[0] - 1))  # excluding the 'Total' row
    else:
        x_labels = categories.copy()
        y_labels = categories.copy()
        
    x_labels.append("Total Predicted")
    y_labels.append("Total Actual")


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cbar=cbar, xticklabels=x_labels, cmap=masked_cmap, yticklabels=y_labels, linewidths=0.5, linecolor="white")
    
    plt.gca().xaxis.tick_top()

    plt.gca().add_patch(plt.Rectangle((cf.shape[1]-1, 0), 1, cf.shape[0], fill=True, color='lightgrey', edgecolor='white', linewidth=0.5))
    plt.gca().add_patch(plt.Rectangle((0, cf.shape[0]-1), cf.shape[1], 1, fill=True, color='lightgrey', edgecolor='white', linewidth=0.5))
    plt.gca().add_patch(plt.Rectangle((2, cf.shape[1]-1), cf.shape[1], 1, fill=True, color='grey', edgecolor='white', linewidth=0.5))

    for i in range(cf.shape[1]):
        plt.gca().add_patch(plt.Rectangle((i, cf.shape[0]-1), 1, 1, fill=False, edgecolor='white', linewidth=0.5))
        plt.gca().add_patch(plt.Rectangle((cf.shape[1]-1, i), 1, 1, fill=False, edgecolor='white', linewidth=0.5))

    if title:
        plt.title(title)
    
    path=  "NewCM"+name+".png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0.5)
    plt.show()


# In[45]:


#from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import log_loss,roc_auc_score,precision_score,roc_curve,auc,matthews_corrcoef
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt


def plot_metrices3(cm,name):
    
    
    
    #total = sum(sum(cm))
    #acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    #print(cm)

    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    #print('Sensitivity (TPR) : ',TPR)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    print('Specificity (TNR) : ',TNR)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    #print('Precision (PPV) : ',PPV)
    # Negative predictive value
    NPV = TN/(TN+FN)
    print('NPV : ',NPV)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print('FPR : ',FPR)
    # False negative rate
    FNR = FN/(TP+FN)
    print('FNR : ',FNR)
    # False discovery rate
    FDR = FP/(TP+FP)
    #print('FDR : ',FDR)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print('ACCURACY : ',ACC)
    #print("Average ACCURACY: {:.4f}".format(ACC))
    #print("Average Sensitivity: {:.4f}".format(sensitivity))
    #print("Average Specificity: {:.4f}".format(specificity))
    DSC=(2*TP)/(2*TP+FP+FN)
    #print('DSC : ',DSC)
    #auc = roc_auc_score(ytrue, ypred)
    
    #roc_auc = round(auc(fpr, tpr), 3)

                  
    model_results =pd.DataFrame([[ACC,TPR,TNR,PPV,DSC]],
               columns = ['Accuracy', 'Sensitivity','Specificity','Precision','DSC'])
    

    #class_report_df=
    model_results['Accuracy']=pd.Series(ACC)
    model_results['Specificity']=pd.Series(TNR)
    model_results['TPR']=pd.Series(TPR)
    model_results['FNR']=pd.Series(FNR)
    model_results['NPV']=pd.Series(NPV)



    # dictionary of lists
    dict = {'Accuracy': ACC, 'Specificity': TNR, 'TPR': TPR ,  'FNR': FNR, 'NPV': NPV}
     
    df = pd.DataFrame(dict)
 
    print(df)
    path= "Accuracy of: "+ name +".csv"

    #df.to_csv(path)

    
    return df


# <a id="model"></a>
# # <center>Create EfficientNetB4 Model</center><a id="callback"></a>

# In[46]:


from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Average, Input, Concatenate, GlobalMaxPooling2D,BatchNormalization

def make_Static_model(model_name):  
    class_count=3
    img_size=(224,224)
    img_shape=(img_size[0], img_size[1], 3)
    lr=0.001
    if model_name == "B0":
        base_model=tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max')
        msg='Created EfficientNet B0 model'
    elif model_name == "B1":
        base_model=tf.keras.applications.efficientnet.EfficientNetB1(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created EfficientNet B1 model'
   
    elif model_name == "B2":
        base_model=tf.keras.applications.efficientnet.EfficientNetB2(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created EfficientNet B2 model'
        
    elif model_name == 'DenseNet121':
        base_model=tf.keras.applications.DenseNet121(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created DenseNet121 model'
        
    elif model_name == 'MobileNet':
        base_model=tf.keras.applications.MobileNet(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max')  
        msg='Created MobileNet  model'
                    
    elif model_name == 'ResNet101V2':
        base_model=tf.keras.applications.ResNet101V2(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created ResNet101V2 model'
        
    elif model_name == 'VGG16':
        base_model=tf.keras.applications.VGG16(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created VGG16 model'
       
    elif model_name == 'InceptionResNetV2':
        base_model=tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created InceptionResNetV2 model'
    
    elif model_name == "B3":
        base_model=tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created EfficientNet B3 model'
    
    elif model_name == "EfficientNetB4":
        base_model=tf.keras.applications.efficientnet.EfficientNetB4(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created EfficientNet B4 model'

  
    elif model_name == 'InceptionV3':
        base_model=tf.keras.applications.InceptionV3(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created InceptionV3 model'
        
    elif model_name == 'Xception':
        base_model=tf.keras.applications.Xception(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created Xception model'
                
    else:
        base_model=tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max')
        msg='Created EfficientNet B7 model'   
   
       
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)

    output = Dense(class_count, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy','AUC']) 

    """
    base_model.trainable=True
    x=base_model.output
    x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
    x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                    bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
    x=Dropout(rate=.4, seed=123)(x)       
    output=Dense(class_count, activation='softmax')(x)
    model=Model(inputs=base_model.input, outputs=output)
    
    model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy','AUC']) 
    """
    #msg=msg + f' with initial learning rate set to {lr}'
    #print_in_color(msg)
    return model



# In[47]:


def make_model(model_name):  
    class_count=3
    img_size=(224,224)
    img_shape=(img_size[0], img_size[1], 3)
    lr=0.001
    if model_name == "EfficientNetB0":
        base_model=tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max')
        msg='Created EfficientNet B0 model'
    elif model_name == "EfficientNetB1":
        base_model=tf.keras.applications.efficientnet.EfficientNetB1(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created EfficientNet B1 model'
   
    elif model_name == "EfficientNetB2":
        base_model=tf.keras.applications.efficientnet.EfficientNetB2(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created EfficientNet B2 model'
        
    elif model_name == 'DenseNet121':
        base_model=tf.keras.applications.DenseNet121(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created DenseNet121 model'
        
    elif model_name == 'MobileNet':
        base_model=tf.keras.applications.MobileNet(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max')  
        msg='Created MobileNet  model'
                    
    elif model_name == 'ResNet101V2':
        base_model=tf.keras.applications.ResNet101V2(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created ResNet101V2 model'
        
    elif model_name == 'VGG16':
        base_model=tf.keras.applications.VGG16(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created VGG16 model'
       
    elif model_name == 'InceptionResNetV2':
        base_model=tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created InceptionResNetV2 model'
    
    elif model_name == "EfficientNetB3":
        base_model=tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created EfficientNet B3 model'
    
    elif model_name == "EfficientNetB4":
        base_model=tf.keras.applications.efficientnet.EfficientNetB4(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created EfficientNet B4 model'

  
    elif model_name == 'InceptionV3':
        base_model=tf.keras.applications.InceptionV3(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created InceptionV3 model'
        
    elif model_name == 'Xception':
        base_model=tf.keras.applications.Xception(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
        msg='Created Xception model'
                
    else:
        base_model=tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max')
        msg='Created EfficientNet B7 model'   
   
       
    base_model.trainable=True
    x=base_model.output
    x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
    x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                    bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
    x=Dropout(rate=.4, seed=123)(x)       
    output=Dense(class_count, activation='softmax')(x)
    model=Model(inputs=base_model.input, outputs=output)
    model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy','AUC']) 
    msg=msg + f' with initial learning rate set to {lr}'
    print_in_color(msg)
    return model,base_model



# In[48]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
best_weights_ph1 = "model0.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(best_weights_ph1, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)


# In[49]:


checkpoint_path = "xception_best.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
my_callbacks2 = [
               tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                               monitor = 'val_accuracy',
                               verbose = 1,
                               save_weights_only=True,
                               save_best_only = True,
                               mode="max"),
              EarlyStopping(monitor='val_accuracy',
                            patience=10,
                            verbose=0),
              ReduceLROnPlateau(monitor='val_accuracy',
                                patience=10,
                                verbose=1)
]


my_callbacks = [
               
              EarlyStopping(monitor='val_accuracy',
                            patience=10,
                            verbose=0),
              ReduceLROnPlateau(monitor='val_accuracy',
                                patience=5,
                                verbose=0)
]




# In[50]:


def plot_loss_accuracy2(history,newaccuray,newloss):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    #epochs = range(len(acc))
    plt.style.use('fivethirtyeight')
    Epochs = [i+1 for i in range(len(acc))]
    plt.figure(figsize= (5, 5))

    plt.plot(Epochs, history.history["accuracy"], 'r', label= 'Training accuracy')
    plt.plot(Epochs, history.history['val_accuracy'], 'g', label= 'Validation accuracy')
   
    #plt.plot(history.history["accuracy"], 'bo--', label="accuracy")
    #plt.plot(history.history['val_accuracy'], 'ro--', label="validation accuracy")
    plt.title('Training and Validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.savefig(newaccuray)
    #plt.figure()
    plt.show()
    
    plt.style.use('fivethirtyeight')

    #plt.plot(history.history["loss"], "bo--", label="loss")
    #plt.plot(history.history["val_loss"], "ro--", label = "validation loss")
    plt.figure(figsize= (5, 5))

    plt.plot(Epochs, history.history["loss"], 'r', label= 'Training loss')
    plt.plot(Epochs, history.history["val_loss"], 'g', label= 'Validation loss')
    
    plt.title('Training and Validation loss')
    plt.ylabel('loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(newloss)
    plt.show()


# <a id="train"></a>
# # <center>Instantiate Callback and Train the Model</center><a id="callback"></a>

# In[51]:


epochs =30
patience= 1 # number of epochs to wait to adjust lr if monitored value does not improve
stop_patience =3 # number of epochs to wait before stopping training if monitored value does not improve
threshold=.9 # if train accuracy is < threshhold adjust monitor accuracy, else monitor validation loss
factor=.5 # factor to reduce lr by
dwell=True # experimental, if True and monitored metric does not improve on current epoch set  modelweights back to weights of previous epoch
freeze=False # if true free weights of  the base model
ask_epoch=30 # number of epochs to run before asking if you want to halt training
batches=40


# <a id="plot"></a>
# # <center>Plot training data, evaluate model on test set and save it</center>

# <a id="predict"></a>
# # <center>Make predictions on test set and generate Confusion Matrix and Classification Report</center>

# ### to test the prediction kernel create a directory with an MRI of a non tumor

# <a id="classify"></a>
# # <center>Use Predictor function to Classify the Input Image</center>

# In[52]:


#test_gen=test_gen_paper


# In[53]:


#FFFFFFF


# In[54]:


pip install openpyxl


# In[55]:


from openpyxl.workbook import Workbook


# In[56]:


model_name = 'B2'
modelB2=make_Static_model(model_name)
#model0.compile(Adamax(lr=.001), loss='categorical_crossentropy', metrics=['accuracy']) 
modelB2.compile(Adamax(learning_rate=.001), loss='categorical_crossentropy', metrics=['accuracy','AUC']) 
historyB2=modelB2.fit(x=train_gen,  epochs=epochs, verbose=0, callbacks=my_callbacks,  validation_data=valid_gen,
              validation_steps=None,  shuffle=False,  initial_epoch=0)

losspath="Static_Loss_B2.png"
accuracypath="Static_Accuracy_B2.png"
plot_loss_accuracy(historyB2,losspath,accuracypath)

preds = modelB2.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)
print(y_pred)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

# Classification report
print(classification_report(test_gen.classes, y_pred, target_names= classes))

print_code=0
name="B2"

model_results=plot_metrices3(cm,name)
model_results

plot_cm3(cm,name)


report_with_auc = class_report(name,
    y_true=test_gen.classes, 
    y_pred=y_pred)
print(report_with_auc)



AllB2Results1 = pd.concat([model_results, report_with_auc], axis=1)
#finalResult2= pd.concat([finalResult1, AllB2Results2], axis=0)
path= name +"Results2.xlsx"
AllB2Results1.to_excel(path)
  
path="Static" + name +"Results.xlsx"
AllB2Results1.to_excel(path)


# In[ ]:


model_name = 'MobileNet'
modelMobileNet=make_Static_model(model_name)
modelMobileNet.compile(Adamax(learning_rate=.001), loss='categorical_crossentropy', metrics=['accuracy','AUC']) 
historyMobileNet=modelMobileNet.fit(x=train_gen,  epochs=epochs, verbose=0, callbacks=my_callbacks,  validation_data=valid_gen,
              validation_steps=None,  shuffle=False,  initial_epoch=0)

losspath="Static_Loss_MobileNet.png"
accuracypath="Static_Accuracy_MobileNet.png"
plot_loss_accuracy(historyMobileNet,losspath,accuracypath)

preds = modelMobileNet.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)
print(y_pred)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

# Classification report
print(classification_report(test_gen.classes, y_pred, target_names= classes))

print_code=0
name="MobileNet"

model_results=plot_metrices3(cm,name)
model_results

plot_cm3(cm,name)



report_with_auc = class_report(name,
    y_true=test_gen.classes, 
    y_pred=y_pred)
print(report_with_auc)



AllB2Results2 = pd.concat([model_results, report_with_auc], axis=1)
finalResult1= pd.concat([AllB2Results1, AllB2Results2], axis=0)
path= name +"Results.xlsx"
AllB2Results2.to_excel(path)


# In[ ]:


model_name = 'ResNet101V2'
modelResNet101V2=make_Static_model(model_name)
modelResNet101V2.compile(Adamax(learning_rate=.001), loss='categorical_crossentropy', metrics=['accuracy','AUC']) 
historyResNet101V2=modelResNet101V2.fit(x=train_gen,  epochs=epochs, verbose=0, callbacks=my_callbacks,  validation_data=valid_gen,
              validation_steps=None,  shuffle=False,  initial_epoch=0)

losspath="Static_Loss_ResNet101V2.png"
accuracypath="Static_Accuracy_ResNet101V2.png"
plot_loss_accuracy(historyResNet101V2,losspath,accuracypath)

preds = modelResNet101V2.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)
print(y_pred)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

# Classification report
print(classification_report(test_gen.classes, y_pred, target_names= classes))

print_code=0
name="ResNet101V2"

model_results=plot_metrices3(cm,name)
model_results

plot_cm3(cm,name)



report_with_auc = class_report(name,
    y_true=test_gen.classes, 
    y_pred=y_pred)
print(report_with_auc)



AllB2Results3 = pd.concat([model_results, report_with_auc], axis=1)
finalResult2= pd.concat([finalResult1, AllB2Results3], axis=0)
path= name +"Results.xlsx"
AllB2Results3.to_excel(path)


# In[ ]:


model_name = 'VGG16'
modelVGG16=make_Static_model(model_name)
modelVGG16.compile(Adamax(learning_rate=.001), loss='categorical_crossentropy', metrics=['accuracy','AUC']) 
historyVGG16=modelVGG16.fit(x=train_gen,  epochs=epochs, verbose=0, callbacks=my_callbacks,  validation_data=valid_gen,
              validation_steps=None,  shuffle=False,  initial_epoch=0)

losspath="Static_Loss_VGG16.png"
accuracypath="Static_Accuracy_VGG16.png"
plot_loss_accuracy(historyVGG16,losspath,accuracypath)

preds = modelVGG16.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)
print(y_pred)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

# Classification report
print(classification_report(test_gen.classes, y_pred, target_names= classes))

print_code=0
name="VGG16"

model_results=plot_metrices3(cm,name)
model_results

plot_cm3(cm,name)


report_with_auc = class_report(name,
    y_true=test_gen.classes, 
    y_pred=y_pred)
print(report_with_auc)




AllB2Results4 = pd.concat([model_results, report_with_auc], axis=1)
finalResult3= pd.concat([finalResult2, AllB2Results4], axis=0)
path= name +"Results.xlsx"
AllB2Results4.to_excel(path)


# In[ ]:


model_name = 'Xception'
modelXception=make_Static_model(model_name)
modelXception.compile(Adamax(learning_rate=.001), loss='categorical_crossentropy', metrics=['accuracy','AUC']) 
historymodelXception=modelXception.fit(x=train_gen,  epochs=epochs, verbose=0, callbacks=my_callbacks,  validation_data=valid_gen,
              validation_steps=None,  shuffle=False,  initial_epoch=0)

losspath="Static_Loss_Xception.png"
accuracypath="Static_Accuracy_Xception.png"
plot_loss_accuracy(historymodelXception,losspath,accuracypath)

preds = modelXception.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)
print(y_pred)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

# Classification report
print(classification_report(test_gen.classes, y_pred, target_names= classes))

print_code=0
name="Xception"

model_results=plot_metrices3(cm,name)
model_results

plot_cm3(cm,name)


report_with_auc = class_report(name,
    y_true=test_gen.classes, 
    y_pred=y_pred)
print(report_with_auc)



AllB2Results5 = pd.concat([model_results, report_with_auc], axis=1)
finalResult4= pd.concat([finalResult3, AllB2Results5], axis=0)
path= name +"Results.xlsx"
AllB2Results5.to_excel(path)


# In[ ]:


finalResult4.to_excel("final paper Static ALL Results.xlsx")


# 

# # Dynamic modesl> ******

# In[57]:


from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint


# In[58]:


model_name = 'EfficientNetB2'
dynamicB2,base_model=make_model(model_name)
callbacks=[LRA(model=dynamicB2,base_model= base_model,patience=patience,stop_patience=stop_patience, threshold=threshold,
                   factor=factor,dwell=dwell, batches=batches,initial_epoch=0,epochs=epochs, ask_epoch=ask_epoch )]
historydynamicB2=dynamicB2.fit(x=train_gen,  epochs=epochs, verbose=1, callbacks=callbacks,  validation_data=valid_gen,
                 shuffle=False,  initial_epoch=0)


losspath="Dynamic_B2_loss.png"
accuracypath=" Dynamic_B2_Accuracy.png"
plot_loss_accuracy(historydynamicB2,losspath,accuracypath)

preds = dynamicB2.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)
print(y_pred)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

# Classification report
print(classification_report(test_gen.classes, y_pred, target_names= classes))

print_code=0
name="Dynamic_B2_"

model_results=plot_metrices3(cm,name)
model_results

plot_cm_Dynamic3(cm,name)
make_confusion_matrix(cm, name,
                      categories= ['glioma', 'meningioma', 'Pituitary'],
                      figsize = (12,10),
                      cbar = False
                          )

report_with_auc = class_report(name,
    y_true=test_gen.classes, 
    y_pred=y_pred)
print(report_with_auc)




DDAllB2Results1 = pd.concat([model_results, report_with_auc], axis=1)
#finalResult2= pd.concat([finalResult1, AllB2Results2], axis=0)
path= name +"Dynamic.xlsx"
DDAllB2Results1.to_excel(path)


# In[ ]:


model_name = 'MobileNet'

dynamicMobileNet,base_model=make_model(model_name)
callbacks=[LRA(model=dynamicMobileNet,base_model= base_model,patience=patience,stop_patience=stop_patience, threshold=threshold,
                   factor=factor,dwell=dwell, batches=batches,initial_epoch=0,epochs=epochs, ask_epoch=ask_epoch )]
historydynamicMobileNet=dynamicMobileNet.fit(x=train_gen,  epochs=epochs, verbose=1, callbacks=callbacks,  validation_data=valid_gen,
                 shuffle=False,  initial_epoch=0)


losspath="Dynamic_MobileNet_loss.png"
accuracypath="Dynamic_MobileNet_Accuracy.png"
plot_loss_accuracy(historydynamicMobileNet,losspath,accuracypath)

preds = dynamicMobileNet.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)
print(y_pred)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

# Classification report
print(classification_report(test_gen.classes, y_pred, target_names= classes))

print_code=0
name="Dynamic_MobileNet_"

model_results=plot_metrices3(cm,name)
model_results

plot_cm_Dynamic3(cm,name)
make_confusion_matrix(cm, name,
                      categories= ['glioma', 'meningioma', 'Pituitary'],
                      figsize = (12,10),
                      cbar = False
                          )


report_with_auc = class_report(name,
    y_true=test_gen.classes, 
    y_pred=y_pred)
print(report_with_auc)



DDAllB2Results2 = pd.concat([model_results, report_with_auc], axis=1)
DDfinalResult1= pd.concat([DDAllB2Results1, DDAllB2Results2], axis=0)
path= name +"Dynamic.xlsx"
DDAllB2Results2.to_excel(path)


# In[ ]:


model_name = 'ResNet101V2'
dynamicResNet101V2,base_model=make_model(model_name)
callbacks=[LRA(model=dynamicResNet101V2,base_model= base_model,patience=patience,stop_patience=stop_patience, threshold=threshold,
                   factor=factor,dwell=dwell, batches=batches,initial_epoch=0,epochs=epochs, ask_epoch=ask_epoch )]
historyResNet101V2=dynamicResNet101V2.fit(x=train_gen,  epochs=epochs, verbose=1, callbacks=callbacks,  validation_data=valid_gen,
                 shuffle=False,  initial_epoch=0)


losspath="Dynamic_ResNet101V2_loss.png"
accuracypath=" Dynamic_ResNet101V2_Accuracy.png"
plot_loss_accuracy(historyResNet101V2,losspath,accuracypath)

preds = dynamicResNet101V2.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)
print(y_pred)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

# Classification report
print(classification_report(test_gen.classes, y_pred, target_names= classes))

print_code=0
name="Dynamic_ResNet101V2_"

model_results=plot_metrices3(cm,name)
model_results

plot_cm_Dynamic3(cm,name)
make_confusion_matrix(cm, name,
                      categories= ['glioma', 'meningioma', 'Pituitary'],
                      figsize = (12,10),
                      cbar = False
                          )
report_with_auc = class_report(name,
    y_true=test_gen.classes, 
    y_pred=y_pred)
print(report_with_auc)

DDAllB2Results3 = pd.concat([model_results, report_with_auc], axis=1)
DDfinalResult2= pd.concat([DDfinalResult1, DDAllB2Results3], axis=0)



path= name +"Dynamic.xlsx"
DDAllB2Results3.to_excel(path)


# In[ ]:


def model_VGG16():
    model_name='VGG19'
    base_model=tf.keras.applications.VGG16(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 
    x=base_model.output
    inputs = tf.keras.Input(shape=img_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(Adamax(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model,base_model


# In[ ]:


model_name = 'VGG16'
dynamicVGG16,base_model=model_VGG16()
callbacks=[LRA(model=dynamicVGG16,base_model= base_model,patience=patience,stop_patience=stop_patience, threshold=threshold,
                   factor=factor,dwell=dwell, batches=batches,initial_epoch=0,epochs=epochs, ask_epoch=ask_epoch )]
historyVGG16=dynamicVGG16.fit(x=train_gen,  epochs=epochs, verbose=1, callbacks=callbacks,  validation_data=valid_gen,
                 shuffle=False,  initial_epoch=0)


losspath="Dynamic_VGG16_loss.png"
accuracypath=" Dynamic_VGG16_Accuracy.png"
plot_loss_accuracy(historyVGG16,losspath,accuracypath)

preds = dynamicVGG16.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)
print(y_pred)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

# Classification report
print(classification_report(test_gen.classes, y_pred, target_names= classes))

print_code=0
name="Dynamic_VGG16_"

model_results=plot_metrices3(cm,name)
model_results

plot_cm_Dynamic3(cm,name)
make_confusion_matrix(cm, name,
                      categories= ['glioma', 'meningioma', 'Pituitary'],
                      figsize = (12,10),
                      cbar = False
                          )
report_with_auc = class_report(name,
    y_true=test_gen.classes, 
    y_pred=y_pred)
print(report_with_auc)



DDAllB2Results4 = pd.concat([model_results, report_with_auc], axis=1)
#DDfinalResult3= pd.concat([DDfinalResult2, DDAllB2Results4], axis=0)
path= name +"Dynamic.xlsx"
DDAllB2Results4.to_excel(path)


# In[ ]:


model_name = 'Xception'
dynamicXception,base_model=make_model(model_name)
callbacks=[LRA(model=dynamicXception,base_model= base_model,patience=patience,stop_patience=stop_patience, threshold=threshold,
                   factor=factor,dwell=dwell, batches=batches,initial_epoch=0,epochs=epochs, ask_epoch=ask_epoch )]
historydynamicXception=dynamicXception.fit(x=train_gen,  epochs=epochs, verbose=1, callbacks=callbacks,  validation_data=valid_gen,
                 shuffle=False,  initial_epoch=0)


losspath="Dynamic_Xception_loss.png"
accuracypath=" Dynamic_Xception_Accuracy.png"
plot_loss_accuracy(historydynamicXception,losspath,accuracypath)

preds = dynamicXception.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)
print(y_pred)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

# Classification report
print(classification_report(test_gen.classes, y_pred, target_names= classes))

print_code=0
name="Dynamic_Xception_"

model_results=plot_metrices3(cm,name)
model_results

plot_cm_Dynamic3(cm,name)
make_confusion_matrix(cm, name,
                      categories= ['glioma', 'meningioma', 'Pituitary'],
                      figsize = (12,10),
                      cbar = False
                          )


report_with_auc = class_report(name,
    y_true=test_gen.classes, 
    y_pred=y_pred)
print(report_with_auc)




DDAllB2Results5 = pd.concat([model_results, report_with_auc], axis=1)
#DDfinalResult4= pd.concat([DDfinalResult3, DDAllB2Results5], axis=0)
path= name +"Dynamic.xlsx"
DDAllB2Results5.to_excel(path)


# In[ ]:


#finalResult4.to_excel("final paper Static ALL Results.csv")
DDfinalResult4.to_excel("final paper dynamic ALL Results.xlsx")


# In[ ]:


get_ipython().system('zip -r FinalPaperFigshareBrainwithNewCM.zip /kaggle/working')
from IPython.display import FileLink
FileLink(r'FinalPaperFigshareBrainwithNewCM.zip')


# In[ ]:


def plot_loss_accuracy(history,newloss,newaccuray):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'fontname':'Arial', 'size':'14'}
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    #epochs = range(len(acc))
    plt.style.use('fivethirtyeight')
    Epochs = [i+1 for i in range(len(acc))]
    #plt.figure(figsize= (15, 15))
    ax.xaxis.label.set_size(20)
    plt.plot(Epochs, history.history["accuracy"], 'r', label= 'Training accuracy')
    plt.plot(Epochs, history.history['val_accuracy'], 'g', label= 'Validation accuracy')
   
    #plt.plot(history.history["accuracy"], 'bo--', label="accuracy")
    #plt.plot(history.history['val_accuracy'], 'ro--', label="validation accuracy")
    ax.set_title('Training and Validation accuracy',fontweight="bold", size=20)
    ax.set_ylabel('accuracy',fontsize=25)
    ax.set_xlabel('Epochs',fontsize=25)
    ax.yaxis.set_tick_params(labelsize='large')
    ax.xaxis.set_tick_params(labelsize='large')
    ax.legend()

    fig.savefig(newaccuray,dpi=1200)
    #plt.figure()
    plt.show()
    
    plt.style.use('fivethirtyeight')

    #plt.plot(history.history["loss"], "bo--", label="loss")
    #plt.plot(history.history["val_loss"], "ro--", label = "validation loss")
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    plt.plot(Epochs, history.history["loss"], 'r', label= 'Training loss')
    plt.plot(Epochs, history.history["val_loss"], 'g', label= 'Validation loss')
    
    ax.set_title('Training and Validation loss')
    ax.set_ylabel('loss',fontsize=20)
    ax.set_xlabel('Epochs',fontsize=20)
    ax.yaxis.set_tick_params(labelsize='large')
    ax.xaxis.set_tick_params(labelsize='large')
    plt.legend()
    fig.savefig(newloss,dpi=1200)
    plt.show()


# In[ ]:


losspath="Static_Loss_B2.png"
accuracypath="Static_Accuracy_B2.png"
plot_loss_accuracy(historyB2,losspath,accuracypath)


# In[ ]:


losspath="Static_Loss_MobileNet.png"
accuracypath="Static_Accuracy_MobileNet.png"
plot_loss_accuracy(historyMobileNet,losspath,accuracypath)


# In[ ]:


losspath="Static_Loss_ResNet101V2.png"
accuracypath="Static_Accuracy_ResNet101V2.png"
plot_loss_accuracy(historyResNet101V2,losspath,accuracypath)


# In[ ]:


losspath="Static_Loss_VGG16.png"
accuracypath="Static_Accuracy_VGG16.png"
plot_loss_accuracy(historyVGG16,losspath,accuracypath)


# In[ ]:


losspath="Static_Loss_Xception.png"
accuracypath="Static_Accuracy_Xception.png"
plot_loss_accuracy(historymodelXception,losspath,accuracypath)


# In[ ]:


losspath="Dynamic_B2_loss.png"
accuracypath=" Dynamic_B2_Accuracy.png"
plot_loss_accuracy(historydynamicB2,losspath,accuracypath)


# In[ ]:


losspath="Dynamic_MobileNet_loss.png"
accuracypath="Dynamic_MobileNet_Accuracy.png"
plot_loss_accuracy(historydynamicMobileNet,losspath,accuracypath)


# In[ ]:


losspath="Dynamic_ResNet101V2_loss.png"
accuracypath=" Dynamic_ResNet101V2_Accuracy.png"
plot_loss_accuracy(historyResNet101V2,losspath,accuracypath)


# In[ ]:


losspath="Dynamic_VGG16_loss.png"
accuracypath=" Dynamic_VGG16_Accuracy.png"
plot_loss_accuracy(historyVGG16,losspath,accuracypath)


# In[ ]:


losspath="Dynamic_Xception_loss.png"
accuracypath=" Dynamic_Xception_Accuracy.png"
plot_loss_accuracy(historydynamicXception,losspath,accuracypath)


# In[ ]:





# In[ ]:


from skimage import data, color, io, img_as_float

def get_heatmap(vgg_conv, processed_image, class_idx):
    # we want the activations for the predicted label
    class_output = vgg_conv.output[:, class_idx]
    
    # choose the last conv layer in your model
    last_conv_layer = vgg_conv.get_layer('block5_conv3')
    
    # get the gradients wrt to the last conv layer
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    
    # we pool the gradients over all the axes leaving out the channel dimension
    pooled_grads = K.mean(grads, axis=(0,1,2))
    
    # Define a function that generates the values for the output and gradients
    iterate = K.function([vgg_conv.input], [pooled_grads, last_conv_layer.output[0]])
    
    # get the values
    grads_values, conv_ouput_values = iterate([processed_image])
    
    # iterate over each feature map in your conv output and multiply
    # the gradient values with the conv output values. This gives an 
    # indication of "how important a feature is"
    for i in range(512): # we have 512 features in our last conv layer
        conv_ouput_values[:,:,i] *= grads_values[i]
    
    # create a heatmap
    heatmap = np.mean(conv_ouput_values, axis=-1)
    
    # remove negative values
    heatmap = np.maximum(heatmap, 0)
    
    # normalize
    heatmap /= heatmap.max()
    
    return heatmap


# In[ ]:


def print_GradCAM(base_model, path_image):
    # select the sample and read the corresponding image and label
    sample_image = cv2.imread(path_image)
    
    # pre-process the image
    sample_image = cv2.resize(sample_image, (300,300))
    #sample_image = cv2.resize(sample_image, None, fx=0.5, fy=0.5)  
    if sample_image.shape[2] ==1:
                sample_image = np.dstack([sample_image, sample_image, sample_image])
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    sample_image = sample_image.astype(np.float32)/255.
#     sample_label = 1

    #since we pass only one image,we expand dim to include batch size 1
    sample_image_processed = np.expand_dims(sample_image, axis=0)

    # get the label predicted by our original model
    pred_label = np.argmax(base_model.predict(sample_image_processed), axis=-1)[0]
    print(base_model.predict(sample_image_processed))

    # get the heatmap for class activation map(CAM)
    heatmap = get_heatmap(base_model, sample_image_processed, pred_label)
    heatmap = cv2.resize(heatmap, (sample_image.shape[0], sample_image.shape[1]))
    heatmap = heatmap *255
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap = 255 - heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # f,ax = plt.subplots(1,2, figsize=(16,6))
    plt.figure()
    f, ax = plt.subplots(ncols=2, figsize=(10, 10))

    #ax[1].imshow(heatmap)
    #ax[1].set_title("heatmap")
    #ax[1].axis('off')

    #superimpose the heatmap on the image    
    sample_image_hsv = color.rgb2hsv(sample_image)
    heatmap = color.rgb2hsv(heatmap)

    alpha=0.7
    sample_image_hsv[..., 0] = heatmap[..., 0]
    sample_image_hsv[..., 1] = heatmap[..., 1] * alpha

    img_masked = color.hsv2rgb(sample_image_hsv)

    # f,ax = plt.subplots(1,2, figsize=(16,6))
    ax[0].imshow(sample_image)
    #ax[0].set_title(f"original image (predicted label: {target_names[pred_label]})")
    f.savefig('Original Image.png')
    ax[0].axis('off')


    ax[1].imshow(img_masked)
    #ax[2].set_title("superimposed image")
    f.savefig('superimposed Image.png')
    #ax[1].axis('off')

    plt.show()


# In[ ]:


outputPath1 = '/kaggle/input/brain-tumor-detection/no/No13.jpg'


# In[ ]:


print_GradCAM(model, outputPath1)


# In[90]:


test_data='/kaggle/input/brain-tumor-mri-dataset/Testing'
filepaths=[]
labels=[]

folds=os.listdir(test_data)

for fold in folds:
    foldpath=os.path.join(test_data,fold)
    filelist=os.listdir(foldpath)
    
    for fpath in filelist:
        fillpath=os.path.join(foldpath,fpath)
        
        labels.append(fold)
        filepaths.append(fillpath)
        
file_series=pd.Series(filepaths,name='filepaths')
label_series=pd.Series(labels,name='labels')
ts_df=pd.concat([file_series,label_series],axis=1)


# In[97]:


ts_df


# In[71]:


def scalar(img):    
    return img  # EfficientNet expects pixelsin range 0 to 255 so no scaling is required


# In[72]:


tvgen=ImageDataGenerator(preprocessing_function=scalar)


# In[128]:


test_gen=tvgen.flow_from_dataframe( df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=False, batch_size=test_batch_size)


# In[136]:


def plot_sample_predictions(model, test_generator, categories, test_dir, num_samples=25, figsize=(15, 20)):
    """
    Nice display of prediction samples to see CNN predictions
    for classification.
    """
    # Make predictions on the test dataset
    predictions = model.predict(test_generator)
    predicted_categories = np.argmax(predictions, axis=1)
    true_categories = test_generator.classes

    # Randomly sample test images
    test_images = np.array(test_generator.filepaths)
    sample_indices = np.random.choice(len(test_images), size=num_samples, replace=False)
    sample_images = test_images[sample_indices]
    sample_predictions = [categories[predicted_categories[i]] for i in sample_indices]
    sample_true_labels = [categories[true_categories[i]] for i in sample_indices]

    # Plot sample images with their predicted and true labels
    plt.figure(figsize=figsize)
    
    # Loop over samples
    for i, image_path in enumerate(sample_images):
        # Form subplot and plot
        plt.subplot(3, 3, i + 1)
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.axis("off")
        
        # Set axis label color depending on correct prediction or not
        prediction_color = 'green' if sample_predictions[i] == sample_true_labels[i] else 'red'
        plt.title(f"Predicted: {sample_predictions[i]}\nTrue: {sample_true_labels[i]}", color=prediction_color)
        
    plt.tight_layout()
    plt.savefig("prediction images.png",dpi=1200)

    plt.show()
# Accessing class indices for training data generator

# Using functions in 6.1 for showing results
plot_sample_predictions(model=dynamicB2, 
                        test_generator=test_gen, 
                        categories=df['labels'],
                        test_dir=test_data, 
                        num_samples=9,
                        figsize=(13, 12))


# In[ ]:


# Accessing class indices for training data generator

# Using functions in 6.1 for showing results
plot_sample_predictions(model=model, 
                        test_generator=test_genrator, 
                        categories=ts_df['labels'],
                        test_dir=test_data, 
                        num_samples=9,
                        figsize=(13, 12))


# In[101]:


from PIL import Image


# In[106]:


from keras.preprocessing import image


# In[121]:


def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the image like during training
    return img_array

def predict_image(img_path):
    img_array = prepare_image(img_path)
    predictions = dynamicB2.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Mapping the class index to class label
    class_labels = {v: k for k, v in test_gen.class_indices.items()}
    predicted_label = class_labels[predicted_class[0]]

    return predicted_label

def display_prediction(img_path):
    predicted_label = predict_image(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_label}')
    plt.axis('off')
    plt.show()


# In[124]:


# Example prediction


img_path = '/kaggle/input/brain-tumor-mri-dataset/Testing/pituitary/Te-pi_0028.jpg'



#ERROR prediction
#img_path = '/kaggle/input/brain-tumor-mri-dataset/Testing/glioma/Te-gl_0011.jpg'
#img_path = '/kaggle/input/brain-tumor-mri-dataset/Testing/glioma/Te-gl_0016.jpg'
#img_path = '/kaggle/input/brain-tumor-mri-dataset/Testing/notumor/Te-no_0010.jpg'
#img_path = '/kaggle/input/brain-tumor-mri-dataset/Testing/pituitary/Te-piTr_0000.jpg'
#img_path = '/kaggle/input/brain-tumor-mri-dataset/Testing/pituitary/Te-pi_0028.jpg'




#correct Prediction
#img_path = '/kaggle/input/brain-tumor-mri-dataset/Testing/glioma/Te-glTr_0001.jpg'
#img_path = '/kaggle/input/brain-tumor-mri-dataset/Testing/glioma/Te-glTr_0009.jpg'
#img_path = '/kaggle/input/brain-tumor-mri-dataset/Testing/meningioma/Te-me_0011.jpg'
#img_path = '/kaggle/input/brain-tumor-mri-dataset/Testing/meningioma/Te-me_0021.jpg'



# Display the prediction
display_prediction(img_path)


# In[135]:


import pandas as pd

# initialize list of lists
data = [['/kaggle/input/brain-tumor-mri-dataset/Testing/glioma/Te-gl_0011.jpg', 'glioma'],
        ['/kaggle/input/brain-tumor-mri-dataset/Testing/glioma/Te-gl_0016.jpg', 'glioma'], 
         ['/kaggle/input/brain-tumor-mri-dataset/Testing/notumor/Te-no_0010.jpg', 'notumor'],
         ['/kaggle/input/brain-tumor-mri-dataset/Testing/pituitary/Te-piTr_0000.jpg', 'pituitary'],
         ['/kaggle/input/brain-tumor-mri-dataset/Testing/pituitary/Te-pi_0028.jpg', 'pituitary'],
         ['/kaggle/input/brain-tumor-mri-dataset/Testing/glioma/Te-glTr_0001.jpg', 'glioma'],
                    ['/kaggle/input/brain-tumor-mri-dataset/Testing/meningioma/Te-me_0011.jpg', 'meningioma'],
        ['/kaggle/input/brain-tumor-mri-dataset/Testing/meningioma/Te-me_0021.jpg', 'meningioma'],
        ['/kaggle/input/brain-tumor-mri-dataset/Testing/glioma/Te-glTr_0009.jpg', 'glioma']
       
       
       ]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['filepaths', 'labels'])

# print dataframe.
print(df)


# <a id="eval"></a>
# # <center>Evaluate Model Performance</center>

# ### The EfficientNetB4 model did well achieving an F1 score on the test set of 100%
# ### there was 1 misclassification in 300 test images
# 
# 
