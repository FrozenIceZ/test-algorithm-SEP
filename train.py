import json
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM as LSTM_LAYER, Dense, TimeDistributed, \
    RepeatVector
from keras.models import load_model


def read_json(filename):
    """ Read the JSON data.

    :param source: root of the provided data
    :param filename: name of the file to read
    :type source: string
    """
    with open(filename, 'r') as file:
        data = json.load(file)

    return data


def write_json(data, filename):
    """ Read data back to a JSON file

    :param data: the data to be read back, can have string or dt object for dates
    :param filename: the name of the file to write to
    :param source: root of the provided data
    """
    with open(filename + '.json', 'w') as file:
        json.dump(data, file)


class LSTM_Autoencoder:
    """
    Simplified example of an LSTM encoder-decoder (if you want some extra explanations about these model, let me know
    and I can search for some good online resources).

    *** SEE MY EMAIL FOR A VERY IMPORTANT REMARK REGARDING DATA NORMALIZATION WHEN TRAINING MODELS ***

    *** YOU WILL NEED TO RE-WRITE THE _get_training_targets METHOD TO MATCH YOUR DATA TO BE ABLE TO EXECUTE THIS ***
    """

    def __init__(self):
        # Normally you would make all these parameters arguments to the constructor to customize your model.
        # I added them here like this for simplicity.
        self.PREDICTION_STEPS = 4  # number of predicted timesteps in the future
        self.OBSERVED_STEPS = 7  # number of observed timesteps given as input
        self.PREDICTION_FEATURES = 2  # normally you just want to predict (x,y,z), but sometimes other features are interesting
        self.OBSERVED_FEATURES = 5  # number of observed features given as input (e.g. x,y,v,...etc.)
        self.NEURONS = 128  # for the model architecture, if you run out of RAM, lower it to 64 or 32 (performance might suffer, but I am not sure)
        self.EPOCHS = 20  # number of epochs for training (you might wanna lower it if using a lot of data)
        self.lr = 1e-3  # learning rate (for training)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr)  # optimizer to use during training
        self.loss_fn = tf.keras.losses.MeanAbsoluteError()  # loss function to optimize (for training)
        self.current_epoch = -1

        self.model = self._init_model(self)

    def _init_model(self):
        observation_frames = self.OBSERVED_STEPS
        num_observation_features = self.OBSERVED_FEATURES
        prediction_frames = self.PREDICTION_STEPS
        num_predicted_features = self.PREDICTION_FEATURES  # only (x,y,z)

        model = Sequential()
        model.add(LSTM_LAYER(self.NEURONS, input_shape=(
        observation_frames, num_observation_features)))
        model.add(RepeatVector(prediction_frames))
        model.add(LSTM_LAYER(self.NEURONS, return_sequences=True))
        model.add(TimeDistributed(Dense(num_predicted_features)))

        return model

    # you normally wanna save the loss (both for training and validation data) for every epoch, to later
    # visualize how the training went
    def _log_epoch(self, epoch):
        path = "StoredData/training_history.txt"
        mode = "w" if epoch == 0 else "a"
        line = "{}\n".format(epoch)

        with open(path, mode) as f:
            f.write(line)

    def predict(self, input_observations):
        """
        Input_observations are the observed features (x,y,v...) for the specified number of timesteps
        This type of model expects input_observations to be a tensor (i.e. multidimensional matrix) of
        shape (None, num_observed_timesteps, num_observed_features), where None is the number of trajectories
        to predict (so it is not fixed, but the other two dimensions are fixed and you will get an error if it
        does not match what you specified when initializing the model)
        """
        return self.model.predict(input_observations)

    # *see my remark about data normalization*
    def train(self, data_train, data_val):
        """
        Here it is assumed that the shape of data_train and data_val are once again tensors.
        These are normally given in batches of trajectories, so the tensor shape would be:
        (num_batches, batch_size, timesteps, num_features)
        """

        for epoch in range(self.EPOCHS):
            self.current_epoch = epoch
            if epoch % 10 == 0 and epoch > 0:  # Optional: save model every 10 epochs
                self.save(self)

            # Iterate over the batches of the training dataset.
            batches, total_loss = 0, 0
            for step, batch in enumerate(data_train):
                loss_value = self.train_step(self, batch)
                batches += 1
                total_loss += loss_value

            #avg_loss_train = float(total_loss) / float(batches)

            # [Optional, but common practice] Also compute loss on validation set (every fewer epochs)
            batches_val, total_loss_val = 0, 0
            for step, batch in enumerate(data_val):
                loss_value = self.val_step(self, batch)
                batches_val += 1
                total_loss_val += loss_value

            #avg_loss_val = float(total_loss_val) / float(batches_val)

            # save loss at this epoch
            self._log_epoch(self, epoch)

    def _get_training_targets(self, inputs):
        """
        This method basically takes the entire trajectories, selects which ones to use for training, and
        splits them into the observed_behavior and the predictions we are trying to teach our models.

        P.S: Depending on the shape of your "inputs" here, you will have to re-write this method... I could not convert
        it to fit your data format. Currently I just simplified it from a model I developed recently. Also watch out for
        some variables that were never initialized (e.g. DOWNSAMPLER)

        """

        states_observed = inputs[:, 0:7,
                          :]  # takes first 7 timesteps of the input
        gt_targets = inputs[:, 7:11,
                     1:3]  # takes last 4 timesteps as the training target and only x,y,z
        # sometimes in the recorded data, the object is not visible, so we do not want to use these timesteps
        # for computing the error (and therefore we set the weights of those steps to 0)

        return states_observed, gt_targets

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            states_observed, gt_targets = self._get_training_targets(self,
                                                                     inputs)
            pred_trajectory = self.model(states_observed, training=True)
            loss_value = self.loss_fn(gt_targets, pred_trajectory)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))

        return loss_value

    def val_step(self, inputs):
        states_observed, gt_targets = self._get_training_targets(self, inputs)

        pred_trajectory = self.model(states_observed, training=False)
        loss_value = self.loss_fn(gt_targets, pred_trajectory)
        return loss_value

    def save(self):
        # you normally wanna save your model after training it, or even every few epochs
        path = "StoredData/trained_model"
        self.model.save(path)
        print("saved trained model to: ", path)

    def load():
        # you normally wanna save your model after training it, or even every few epochs
        path = "StoredData/trained_model"
        network = load_model(path)
        print("loaded trained model from: ", path)
        return network


def training():
    allData = read_json("data.txt")  # loads data
    allDataPathsTensor = tf.zeros([0, 11, 5])  # Inits tensor in correct shape
    for scenes in range(
            len(allData)):  # loops through json data to convert into tensor of correct shape
        worldData = allData[scenes]["world"]
        worldData = json.loads(worldData)
        scenePathsTensor = tf.zeros([0, 11, 5])
        for agent in worldData:
            agent_movement = agent["motion"]
            agentPathTensor = tf.zeros([0, 11, 5])
            observedSteps = 11
            counter = 0
            agentStepsTensor = tf.zeros([0, 5])
            for s in range(len(agent_movement)):
                if counter == observedSteps:
                    agentPathTensor = tf.concat(
                        [agentPathTensor, [agentStepsTensor]],
                        axis=0)  # make paths of 11 steps
                    agentStepsTensor = tf.zeros([0, 5])
                    counter = 0

                positionTensor = tf.constant(  # store agent step in tensor
                    [agent_movement[s]['timestamp'],
                     agent_movement[s]['position']['pos_x'],
                     agent_movement[s]['position']['pos_y'],
                     agent_movement[s]['velocity']['vel_x'],
                     agent_movement[s]['velocity']['vel_y']])
                agentStepsTensor = tf.concat(
                    [agentStepsTensor, [positionTensor]], axis=0)
                counter += 1

            scenePathsTensor = tf.concat([scenePathsTensor, agentPathTensor],
                                         axis=0)  # adds all paths together
        allDataPathsTensor = tf.concat([allDataPathsTensor, scenePathsTensor],
                                       axis=0)

    allDataPathsmax = tf.reduce_max(allDataPathsTensor,
                                    axis=1)  # calculates max and min of all attributes
    allDataPathsmax2 = tf.reduce_max(allDataPathsmax, axis=0)
    allDataPathsmin = tf.reduce_min(allDataPathsTensor, axis=1)
    allDataPathsmin2 = tf.reduce_min(allDataPathsmin, axis=0)

    allDataPathsTensorNormalized = tf.keras.utils.normalize(allDataPathsTensor,
                                                            axis=2,
                                                            order=1)  # normalize data

    allDataPathsmaxNormalized = tf.reduce_max(allDataPathsTensorNormalized,
                                              axis=1)  # calculates max and min of all normalized attributes
    allDataPathsmax2Normalized = tf.reduce_max(allDataPathsmaxNormalized,
                                               axis=0)
    allDataPathsminNormalized = tf.reduce_min(allDataPathsTensorNormalized,
                                              axis=1)
    allDataPathsmin2Normalized = tf.reduce_min(allDataPathsminNormalized,
                                               axis=0)

    np.save("StoredData/allDataPathsmax2.npy", allDataPathsmax2,
            allow_pickle=False)  # stores the normalization tensors
    np.save("StoredData/allDataPathsmin2.npy", allDataPathsmin2,
            allow_pickle=False)
    np.save("StoredData/allDataPathsmax2Normalized.npy",
            allDataPathsmax2Normalized, allow_pickle=False)
    np.save("StoredData/allDataPathsmin2Normalized.npy",
            allDataPathsmin2Normalized, allow_pickle=False)

    size = allDataPathsTensorNormalized.get_shape().as_list()  # splits data into training and validation set
    splitValue = math.ceil(size[0] * 0.8)
    data_train_whole, data_val_whole = tf.split(allDataPathsTensorNormalized,
                                                [splitValue,
                                                 (size[0] - splitValue)], 0)

    i = 0
    data_train = tf.zeros([0, 20, 11, 5])  # inits Tensor in correct shape
    while i < splitValue:  # splits data train into batches of size 20
        data_train_segment = data_train_whole[i: i + 20, :, :]
        if (splitValue - i) < 19:
            break
        data_train = tf.concat([data_train, [data_train_segment]], axis=0)
        i += 20
    i = 0
    data_val = tf.zeros([0, 20, 11, 5])  # inits Tensor in correct shape
    while i < splitValue:  # splits data val into batches of size 20
        data_val_segment = data_val_whole[i: i + 20, :, :]
        if (size[0] - splitValue - i) < 19:
            break
        data_val = tf.concat([data_val, [data_val_segment]], axis=0)
        i += 20

    network = LSTM_Autoencoder  # create network
    network.__init__(network)  # init network
    network.train(network, data_train, data_val)  # train network
    network.save(network)  # save network


def main():
    mypath = "StoredData"  # create folder if not present
    if not os.path.isdir(mypath):
        os.makedirs(mypath)

    training()
    # predicting()


if __name__ == '__main__':
    main()
