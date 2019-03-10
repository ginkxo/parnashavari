# fine-tuning InceptionV3 on a new set of classes 
# try replacing this with InceptionV4 

# IMPORTS 

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Concatenate
from keras import backend as K

# ======== MODEL CORE ==========

# values 

X = 1
Y = 1 
NUM_TOOL = 7 
NUM_PHASE = 10 

# custom input size

input_tensor = Input(shape=(X, Y, 3))

# create the base pre-trained model
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)

# rest of model

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# two layers:
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)

# tool prediction layer:
tool_pred = Dense(NUM_TOOL, activation='softmax')(x)

# phase prediction layer:
# concatenate tool_pred to last x 
conc = Concatenate([tool_pred, x]) # we are probably going to use this as our feature map inputs ... ?

# softmax phase prediction 
phase_pred = Dense(NUM_PHASE, activation='softmax')(conc)

tool_model = Model(inputs=base_model.input, outputs=tool_pred)
phase_model = Model(inputs=base_model.input, outputs=phase_pred)

# ======= TRAINING BASES =======

# training the top layers only 
for layer in base_model.layers:
    layer.trainable = False 

phase_model.compile()
phase_model.fit_generator()

tool_model.compile()
tool_model.fit_generator()

for i, layer in enumerate(base_model.layers):
    print(i, layer_name)

# train top 2 inception blocks:
# freeze first 249 layers 
# unfreeze rest

for layer in phase_model.layers[:249]:
    layer.trainable = False 
for layer in phase_model.layers[249:]:
    layer.trainable = True 


for layer in tool_model.layers[:249]:
    layer.trainable = False 
for layer in tool_model.layers[249:]:
    layer.trainable = True 

phase_model.compile()
tool_model.compile()

phase_model.fit_generator()
tool_model.fit_generator() 






