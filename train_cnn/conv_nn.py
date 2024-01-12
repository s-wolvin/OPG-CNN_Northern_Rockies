""" 
Savanna Wolvin
Created: Sep 27th, 2022
Edited: Jun 1st, 2023
    

##### SUMMARY #####
Script file holding the functions to create, train, and test a Convolutional 
Neural Network for cnn_regional_facet_MAIN.py

##### FUNCTION LIST ##########################################################
    train_cnn() - Function to train the CNN model, plots training and 
                    validation loss, heatmaps of actual VS predicted for 
                    training and validation datasets
    structure_cnn() - Function to create the structure of the CNN



"""
#%% Global Imports

import numpy as np
import cnn_plots as plots
import tensorflow.keras as tf_k
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd




#%%
""" CLASS DEFINITION: weighted_loss
    FUNCTIONS
    mae - Mean Absolute Error
    mse - Mean Squared Error

    INPUTS
    true - Observed OPG Values
    pred - Predicted OPG Values
    mask - 
    
    OUTPUT () - 
"""

class weighted_MSE(tf_k.losses.Loss):
    def call(self, y_true, y_pred):
        se  = tf.math.square(y_pred - y_true)
        mask = tf.cast(y_true != 0, tf.float32)
        weight_se = se * mask
        weight_mse = tf.reduce_mean(weight_se, axis=-1)
        return weight_mse
    
class weighted_MAE(tf_k.losses.Loss):
    def call(self, y_true, y_pred):
        ae  = tf.math.abs(y_pred - y_true)
        mask = tf.cast(y_true != 0, tf.float32)
        weight_ae = ae * mask
        weight_mae = tf.reduce_mean(weight_ae, axis=-1)
        return weight_mae




#%% 
""" FUNCTION DEFINITION: train_cnn
    INPUTS
    model           - Unfitted ConvNet
    opg_type        - Identifier For What Type of OPG Values Is Used In Training
    save_dir        - Directory to Save the Figure
    epoch_num       - Max Number of Epochs to Train the Neural Network
    patienceX       - Number of Epochs With No Improvement to Stop Training
    optimizer_class - Type of Optimizer used in Training
    batch_sz        - Size of the Batch Used During Training Epochs 
    loss_metric     - Metric Used to Measure Loss
    atmos_train     - Atmospheric data to train the model
    opg_train       - OPG data to train the model
    atmos_vldtn     - Atmospheric data to validate the model at every iteration
    opg_vldtn       - OPG data to validate the model at every iteration
        
    OUTPUT (model) - New trained model based on the previous structure
"""
    
def train_cnn(model, opg_type, save_dir, epoch_num, patienceX, batch_sz, 
              optimizer_class, loss_metric, atmos_train, opg_train, atmos_vldtn, 
              opg_vldtn):  
    print('Train the CNN...')
    
    opg_trainX = np.array(opg_train['opg'])

    # Calculate Size of Data
    atm_sz = dict(atmos_train.sizes)
    opg_sz = dict(opg_train.sizes)
    count_channels = 0
    count_on_facet = 0
    for ii in atmos_train.values():
        if len(ii.dims) == 3: count_channels += 1
        elif len(ii.dims) == 2: count_on_facet += 1
    
    # Create Empty Array for Training Data
    atmos_train_4D_X = np.zeros((atm_sz['time'], atm_sz['lat'], 
                                 atm_sz['lon'], count_channels))
    atmos_train_OF_X = np.zeros((atm_sz['time'], opg_sz['facet_num']*count_on_facet))
    
    
    # Pull Variables 
    count_4D = 0
    count_OF = 0
    for var_name, values in atmos_train.items():
        if len(values.dims) == 3:
            atmos_train_4D_X[:,:,:,count_4D] = np.array(values)
            count_4D += 1
        elif len(values.dims) == 2: 
            atmos_train_OF_X[:,opg_sz['facet_num']*count_OF:opg_sz['facet_num']*(count_OF+1)] = np.array(values)
            count_OF += 1
    
    # Compile CNN Model
    model.compile(optimizer="rmsprop", loss=weighted_MSE(),
                  metrics = ["mean_squared_error", "mean_absolute_error"])
    
    # Combine Inputs if On-Facet Data Exists
    if count_on_facet > 0: inputs = [atmos_train_4D_X, atmos_train_OF_X]
    else: inputs = [atmos_train_4D_X]
    
    atm_sz_val = dict(atmos_vldtn.sizes)
    opg_sz_val = dict(opg_vldtn.sizes)   
    
    # Train the CNN and Plot Loss
    if atm_sz_val['time'] != 0:
        # Create Empty Array for Validation Data
        atmos_vldtn_4D_X = np.zeros((atm_sz_val['time'], atm_sz_val['lat'], 
                                     atm_sz_val['lon'], count_channels))
        atmos_vldtn_OF_X = np.zeros((atm_sz_val['time'], opg_sz_val['facet_num']*count_on_facet))
        
        # Pull Variables for Validation
        count_4D = 0
        count_OF = 0
        for var_name, values in atmos_vldtn.items():
            if len(values.dims) == 3:
                atmos_vldtn_4D_X[:,:,:,count_4D] = np.array(values)
                count_4D += 1
            elif len(values.dims) == 2: 
                atmos_vldtn_OF_X[:,opg_sz_val['facet_num']*count_OF:opg_sz_val['facet_num']*(count_OF+1)] = np.array(values)
                count_OF += 1
        opg_vldtnX = np.array(opg_vldtn['opg'])
        
        # Combine Inputs if On-Facet Data Exists
        if count_on_facet > 0: inputs_vldtn = [atmos_vldtn_4D_X, atmos_vldtn_OF_X]
        else: inputs_vldtn = [atmos_vldtn_4D_X]
        
        # Define callback
        callback = [EarlyStopping(monitor='val_loss', patience=patienceX, mode='min')]
    
        history = model.fit(inputs, opg_trainX, 
                            epochs = epoch_num, 
                            batch_size = batch_sz, 
                            validation_data=(inputs_vldtn, opg_vldtnX),
                            shuffle = True,
                            callbacks=[callback])
        
        plots.training_validation_loss(save_dir, history.history, loss_metric, opg_type)

    else:
        # Define callback
        callback = EarlyStopping(monitor='loss', patience=patienceX, mode='min')
        
        history = model.fit(inputs, opg_trainX, 
                            epochs = epoch_num, 
                            batch_size = batch_sz,
                            shuffle = True,
                            callbacks=[callback])
        plots.training_loss(save_dir, history, loss_metric, opg_type)
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(save_dir + "history.csv")
    
    return model

    
    
    
#%% 
""" FUNCTION DEFINITION: structure_cnn
    INPUTS
    atmosphere      - The Atmospheric Input Data
    conv_act_func   - Activation Function for the Convolutions
    nn_layer_width  - Number of Nodes in Each Hidden Layer
    nn_hidden_layer - Number of Hidden Layers in the Neural Network
    dense_act_func  - Activation Function for the Neural Network
    dropout_rate    - Percent of Randomly Dropped Nodes
    kernal_sz       - Size of the Kernal Window in the Convolutions
    kernal_num      - Number of Kernals
    output_width    - Number of Needed Output Nodes, One for Each Facet
    path            - Directory to save the model figure
        
    OUTPUT (model) - Structure of the CNN
"""

def structure_cnn(atmosphere, conv_act_func, nn_layer_width, nn_hidden_layer,
                  dense_act_func, dropout_rate, kernal_sz, kernal_num, 
                  output_width, path):
    print("Create CNN Structure...")
    
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Calculate Size of Data
    data_coords = dict(atmosphere.sizes)
    count_channels = 0
    count_on_facet = 0
    
    for ii in atmosphere.values():
        if len(ii.dims) == 3: count_channels += 1
        elif len(ii.dims) == 2: count_on_facet += 1
        
    # Define Size of Image Inputs
    inputs = tf_k.Input(shape = (data_coords["lat"], data_coords["lon"], count_channels), name="input")
    x = tf_k.layers.Conv2D(filters=kernal_num, kernel_size=kernal_sz, activation=conv_act_func, padding="same", name="conv2d_1") (inputs)
    x = tf_k.layers.MaxPooling2D(pool_size = 2, name="max_pooling_1")(x)
    x = tf_k.layers.Conv2D(filters=(kernal_num*2), kernel_size=kernal_sz, activation=conv_act_func, padding="same", name="conv2d_2") (x)
    x = tf_k.layers.MaxPooling2D(pool_size = 2, name="max_pooling_2")(x)
    x = tf_k.layers.Conv2D(filters=(kernal_num*4), kernel_size=kernal_sz, activation=conv_act_func, padding="same", name="conv2d_3") (x)
    x = tf_k.layers.MaxPooling2D(pool_size = 2, name="max_pooling_3")(x)

    x = tf_k.layers.Flatten(name="flatten")(x)
    
    # If There is On-Facet Data - Add it to the Model
    if count_on_facet > 0:
        input_length = count_on_facet * np.shape(atmosphere['facet_num'])[0]
        inputs2 = tf_k.layers.Input(shape=(input_length,), name="of_input")
        x = tf_k.layers.concatenate([x, inputs2])

    ## ADDING DROPOUT HERE CAN...
    x = tf_k.layers.Dropout(dropout_rate, name="dropout")(x)
    
    # Create Neural Network ## ADDING DROPOUT TO THE DENSE LAYERS NEGATIVELY AFFECTS PREDICTABILITY
    for hl in range(nn_hidden_layer):
        x = tf_k.layers.Dense(nn_layer_width, activation=dense_act_func, name="dense_"+str(hl))(x)
        
    outputs = tf_k.layers.Dense(output_width, name="predictions") (x)
    
    if count_on_facet > 0:
        model = tf_k.Model(inputs = [inputs, inputs2], outputs = outputs)
    else:
        model = tf_k.Model(inputs = inputs, outputs = outputs)
    
    model.summary()
    tf_k.utils.plot_model(model, to_file=f"{path}model.png", dpi=200, show_shapes=True)
    
    return model



