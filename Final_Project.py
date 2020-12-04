import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape
from tensorflow.keras import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow.keras.backend as K
from Squeeze_and_Excite import Squeeze_and_Excite 
from ResNet50 import *
from LoadDataset import *

epoch = 1
batch_size = 32
learning_rate = 0.01


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    data_X = dict[b'data']

    test_labels = dict[b'labels']
    return data_X, test_labels

def oneHotEncoder(data):
    label_encoder = LabelEncoder()
    values = np.array(data)
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = label_encoder.fit_transform(values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded
"""
def normalize(data, mean_x, std_x):
    data_norm = data / np.max(data)
    data_norm = data_norm - mean_x
    data_norm = data_norm / std_x

    return data_norm
"""

class layerModel(Model):
    def __init__(self, inp_shape):
        super(layerModel, self).__init__()
        self.squeeze = Squeeze_and_Excite(inp_shape, 3) # ratio = 3 
        self.reshape = Flatten()  #Reshape((3072, 1))
        self.h = Dense(100, activation = 'relu', kernel_initializer = 'glorot_uniform', input_shape=(inp_shape, ))
        self.p = Dense(10, activation ='softmax')

    def call(self, x):
        y = self.squeeze(x)
        y = self.reshape(y)
        y = self.h(y)
        y = self.p(y)
        return y

#model = layerModel(3)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training = True)
        print(predictions.get_shape())
        loss =loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)
        
@tf.function
def test_step(images, labels):
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training = False)
    t_loss =loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


if __name__ == "__main__":

    save_model_dir = ".\checkpoints2"

    print("loading data...")
    train_X, train_lab, test_X, test_lab = get_data()
    print("normalizing data...")
    train_X, test_X = normalize(train_X, test_X)
    train_data = tf.data.Dataset.from_tensor_slices((train_X, train_lab)).batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((test_X, test_lab)).batch(10000)

    #Load data
    print("loading data...")
    """
    filetrain1 = "C:\cifar-10-batches-py\data_batch_1"
    filetrain2 = "C:\cifar-10-batches-py\data_batch_2"
    filetrain3 = "C:\cifar-10-batches-py\data_batch_3"
    filetrain4 = "C:\cifar-10-batches-py\data_batch_4"
    filetrain5 = "C:\cifar-10-batches-py\data_batch_5"
    filetest = "C:\cifar-10-batches-py\\test_batch"

    train_X, train_lab = unpickle(filetrain1)
    train_X2, train_lab2 = unpickle(filetrain2)
    train_X3, train_lab3 = unpickle(filetrain3)
    train_X4, train_lab4 = unpickle(filetrain4)
    train_X5, train_lab5 = unpickle(filetrain5)
    train_X = np.concatenate([train_X, train_X2, train_X3, train_X4, train_X5], axis = 0)
    train_lab.extend(train_lab2)
    train_lab.extend(train_lab3)
    train_lab.extend(train_lab4)
    train_lab.extend(train_lab5)

    val_X = train_X[train_X.shape[0]-1000:,:]
    train_X = train_X[:train_X.shape[0] - 1000, :]
    val_lab = train_lab[len(train_lab)-1000:]
    train_lab = train_lab[:len(train_lab)-1000]
    test_X, test_lab = unpickle(filetest)
    """
    """
    #Normalization of data
    print("normalizing data...")
    mean_X = np.mean(train_X / np.max(train_X))
    std_X = np.std(train_X / np.max(train_X))

    
    ohd = oneHotEncoder(train_lab)
    ohd_val = oneHotEncoder(val_lab)
    ohd_test = oneHotEncoder(test_lab)
    
    normTrain = normalize(train_X, mean_X, std_X)
    normVal = normalize(val_X, mean_X, std_X)
    normTest = normalize(test_X, mean_X, std_X)
   

    mean_X = tf.cast(mean_X, tf.float32)
    std_X = tf.cast(std_X, tf.float32)

    #Data format
    train_X = train_X.reshape(len(train_X), 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    val_X = val_X.reshape(len(val_X), 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    test_X = test_X.reshape(len(test_X), 3, 32, 32).transpose(0,2,3,1).astype("uint8")
   
    #Transform data into Dataset for tensor
    print("transforming data...")
    
    train_data = tf.data.Dataset.from_tensor_slices((train_X, train_lab))
    train_data = train_data.map(lambda img, label: ((tf.cast(img/tf.math.reduce_max(img), tf.float32) - mean_X)/std_X, tf.cast(label, tf.int32))).batch(32)
    #tf.reduce_max(random_int_var)    
    
    val_data = tf.data.Dataset.from_tensor_slices((val_X, val_lab))
    val_data = val_data.map(lambda x, label: ((tf.cast(x/tf.math.reduce_max(x), tf.float32) - mean_X) / std_X, tf.cast(label, tf.int32))).batch(1000)

    test_data = tf.data.Dataset.from_tensor_slices((test_X, test_lab))
    test_data = test_data.map(lambda x, label: ((tf.cast(x/tf.math.reduce_max(x), tf.float32) - mean_X) / std_X, tf.cast(label, tf.int32))).batch(10000)
    """

    #model = ResNet50(include_top=True, weights=None, squeeze=False, squeeze_type='Normal')#, input_tensor = tf.data.Dataset.from_tensor_slices((train_X)))
    #input_tensor = tf.placeholder(tf.float32, shape = [None, train_X.shape[0], train_X.shape[1], train_X.shape[2]])
    model = ResNet50(include_top=True, squeeze=True, squeeze_type='identity') #pre, identity, normal

    """
    features, label = iter(train_dataset).next()
    print("example features:", features[0])
    print("example label:", label[0])
    """

    checkpoint_dir = os.path.join(save_model_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restaurado de {}".format(manager.latest_checkpoint))
    else:
        print("Inicializando desde cero")


    losses = []
    accs = []

    for i in range(epoch):
        for images, labels in train_data:
            print(tf.shape(images))
            train_step(images, labels)

            
        #for images, labels in val_data:
            #train_step(images, labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(i+1,
                        train_loss.result(),
                        train_accuracy.result()*100))

        save_path = manager.save()

        losses.append(train_loss.result())
        accs.append(train_accuracy.result())

        # Reinicia las metricas para el siguiente epoch.
        train_loss.reset_states()
        train_accuracy.reset_states()



    for images, labels in test_data:
        test_step(images, labels)

    template = 'Test Loss: {}, Test Accuracy: {}'
    print(template.format(
                        test_loss.result(),
                        test_accuracy.result()*100))

    t = np.linspace(1, epoch, num=epoch)
    plot1 = plt.figure(1)
    plt.plot(t, losses, 'b')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('Training loss evolution')
    plt.savefig('.\Result_Pics\loss_train')
    plot2 = plt.figure(2)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.plot(t, accs, 'r')
    plt.title('Training accuracy evolution')
    plt.savefig('.\\Result_Pics\\acc_train')
    plt.show()

   

    a = 1