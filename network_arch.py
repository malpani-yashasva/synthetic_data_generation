
from tensorflow import keras
from keras import layers
import json

with open("data\metadata.json" 'r') as infile:
    metadata = json.load(infile)
num_labels = metadata['num_labels']
num_features = metadata['num_features']
LATENT_DIMENSIONS = 100


# generator network
# each discrete variable in our dataset is dealt with a softmax output with the size of the number of unique values that discrete variable can take
# at the end all outputs are merged and a new dataset is produced by the generator
InputA = layers.Input(shape=(LATENT_DIMENSIONS + num_labels, ))
a = layers.Dense(50, activation='linear')(InputA)
a = layers.Dense(50, activation='tanh')(a)
a = layers.LeakyReLU(0.3)(a)
a = layers.Dense(25, activation='linear')(a)
a = layers.Dense(25, activation='tanh')(a)
a = layers.LeakyReLU(0.3)(a)
a = layers.Dense(10, activation='linear')(a)
a = layers.Dense(10, activation='tanh')(a)
a = layers.LeakyReLU(0.3)(a)
outputA = layers.Dense(3, activation='softmax')(a)
modelA = keras.Model(inputs=InputA, outputs=outputA)
InputB = layers.Input(shape=(LATENT_DIMENSIONS + num_labels, ))
b = layers.Dense(100, activation='linear')(InputB)
b = layers.Dense(100, activation='tanh')(b)
b = layers.LeakyReLU(0.3)(b)
b = layers.Dense(80, activation='linear')(b)
b = layers.Dense(80, activation='tanh')(b)
b = layers.LeakyReLU(0.3)(b)
b = layers.Dense(50, activation='linear')(b)
b = layers.Dense(50, activation='tanh')(b)
b = layers.LeakyReLU(0.3)(b)
outputB = layers.Dense(66, activation='softmax')(b)
modelB = keras.Model(inputs=InputB, outputs=outputB)
InputC = layers.Input(shape=(LATENT_DIMENSIONS + num_labels, ))
c = layers.Dense(50, activation='linear')(InputC)
c = layers.Dense(50, activation='tanh')(c)
c = layers.LeakyReLU(0.3)(c)
c = layers.Dense(30, activation='linear')(c)
c = layers.Dense(30, activation='tanh')(c)
c = layers.LeakyReLU(0.3)(c)
c = layers.Dense(25, activation='linear')(c)
c = layers.Dense(25, activation='tanh')(c)
c = layers.LeakyReLU(0.3)(c)
c = layers.Dense(15, activation='tanh')(c)
c = layers.LeakyReLU(0.3)(c)
outputC = layers.Dense(11, activation='softmax')(c)
modelC = keras.Model(inputs=InputC, outputs=outputC)

InputE = layers.Input(shape=(LATENT_DIMENSIONS + num_labels, ))
e = layers.Dense(50, activation='linear')(InputE)
e = layers.Dense(50, activation='tanh')(e)
e = layers.LeakyReLU(0.3)(e)
e = layers.Dense(45, activation='linear')(e)
e = layers.Dense(45, activation='tanh')(e)
e = layers.LeakyReLU(0.3)(e)
e = layers.Dense(40, activation='linear')(e)
e = layers.Dense(40, activation='tanh')(e)
e = layers.LeakyReLU(0.3)(e)
outputE = layers.Dense(36, activation='sigmoid')(e)
modelE = keras.Model(inputs=InputE, outputs=outputE)

main_input1 = layers.Input(shape=(LATENT_DIMENSIONS, ))
main_input2 = layers.Input(shape=(num_labels, ))
main_input = layers.Concatenate(axis=1)([main_input1, main_input2])

d = layers.Dense(50, activation='linear')(main_input)
d = layers.Dense(40, activation = 'linear')(d)
d = layers.LeakyReLU(0.3)(d)
d = layers.Dense(30, activation = 'linear')(d)
outputD = layers.Dense(23, activation='softmax')(d)
modelD = keras.Model(inputs=main_input, outputs = outputD)

main_output = layers.Concatenate(axis = 1)([modelE(main_input), modelA(main_input), modelB(main_input), modelC(main_input)])
generator = keras.Model(inputs = [main_input1, main_input2], outputs = [main_output, outputD])


# critic network
# critic network scores each output produced by the generator
critic_input = layers.Input(shape=(num_features, ))
critic_label_input = layers.Input(shape=(num_labels, ))
c1 = layers.Concatenate(axis = 1)([critic_input, critic_label_input])
c = layers.Dense(139, activation='linear')(c1)
c = layers.LeakyReLU(0.3)(c)
c = layers.Dense(139, activation='tanh')(c)
c = layers.Add()([c1, c])
c2 = layers.Dense(120, activation='linear')(c)
c = layers.LeakyReLU(0.3)(c2)
c = layers.Dense(120, activation='tanh')(c)
c = layers.Add()([c2, c])
c3 = layers.Dense(80, activation='linear')(c)
c = layers.LeakyReLU(0.3)(c3)
c = layers.Dense(80, activation='tanh')(c)
c = layers.Add()([c3, c])
c4 = layers.Dense(40, activation='linear')(c)
c = layers.LeakyReLU(0.3)(c4)
c = layers.Dense(40, activation='tanh')(c)
c = layers.Add()([c4, c])
c5 = layers.Dense(10, activation='linear')(c)
c = layers.LeakyReLU(0.3)(c5)
c = layers.Dense(5, activation='linear')(c)
c = layers.LeakyReLU(0.3)(c)
c_output = layers.Dense(1, activation='linear')(c)
critic = keras.Model(inputs = [critic_input, critic_label_input], outputs = c_output) 

