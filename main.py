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

    @staticmethod
    def load():
        # you normally wanna save your model after training it, or even every few epochs
        path = "StoredData/trained_model"
        network = load_model(path)
        print("loaded trained model from: ", path)
        return network



def predicting():
    network = LSTM_Autoencoder.load()  # loads saved model

    allDataPathsmax2 = np.load(
        "StoredData/allDataPathsmax2.npy")  # loads in tensors to normalize and denormalize data
    allDataPathsmin2 = np.load("StoredData/allDataPathsmin2.npy")
    allDataPathsmax2Normalized = np.load(
        "StoredData/allDataPathsmax2Normalized.npy")
    allDataPathsmin2Normalized = np.load(
        "StoredData/allDataPathsmin2Normalized.npy")
    allDataPathsmax2 = tf.convert_to_tensor(allDataPathsmax2)
    allDataPathsmin2 = tf.convert_to_tensor(allDataPathsmin2)
    allDataPathsmax2Normalized = tf.convert_to_tensor(
        allDataPathsmax2Normalized)
    allDataPathsmin2Normalized = tf.convert_to_tensor(
        allDataPathsmin2Normalized)
    allDataPathsDiff = tf.math.subtract(allDataPathsmax2, allDataPathsmin2)
    allDataPathsDiffNormalized = tf.math.subtract(allDataPathsmax2Normalized,
                                                  allDataPathsmin2Normalized)
    normalizeTensor = tf.math.divide(allDataPathsDiffNormalized,
                                     allDataPathsDiff)
    unNormalizeTensor = tf.math.divide(allDataPathsDiff,
                                       allDataPathsDiffNormalized)
    value_not_nan1 = tf.dtypes.cast(
        tf.math.logical_not(tf.math.is_nan(normalizeTensor)), dtype=tf.float32)
    normalizeTensor = tf.math.multiply_no_nan(normalizeTensor, value_not_nan1)
    value_not_nan2 = tf.dtypes.cast(
        tf.math.logical_not(tf.math.is_nan(unNormalizeTensor)),
        dtype=tf.float32)
    unNormalizeTensor = tf.math.multiply_no_nan(unNormalizeTensor,
                                                value_not_nan2)
    unNormalizeTimeTensor = unNormalizeTensor[0:1]
    unNormalizeXTensor = unNormalizeTensor[1:2]
    unNormalizeYTensor = unNormalizeTensor[2:3]
    unNormalizeTime = unNormalizeTimeTensor.numpy()[0]
    unNormalizeX = unNormalizeXTensor.numpy()[0]
    unNormalizeY = unNormalizeYTensor.numpy()[0]   # stores them in proper individual numpy floats

    allData = read_json("data.txt")  # loops through prediction input
    worldData = allData[0]["world"]
    worldData = json.loads(worldData)
    predictionData = {"predictionData": []}
    predictionDict = {"dataset": "DontKnowInCurrentJSON",
                      "scene_id": allData[0]["scene_id"], "agents": []}
    for agent in worldData:
        agent_movement = agent["motion"]
        if len(agent_movement) < 7:
            break
        agentDict = {"agent_id": agent["agent_id"],
                     "initial_timestamp": agent_movement[7]["timestamp"],
                     "predicted_paths": []}
        timeArray = [None] * len(agent_movement)
        agentPathTensor = tf.zeros([0, 5])
        for s in range(len(agent_movement)):
            timeArray[s] = agent_movement[s]['timestamp']
            positionTensor = tf.constant(
                [agent_movement[s]['timestamp'],
                 agent_movement[s]['position']['pos_x'],
                 agent_movement[s]['position']['pos_y'],
                 agent_movement[s]['velocity']['vel_x'],
                 agent_movement[s]['velocity']['vel_y']])
            positionTensor = tf.math.multiply(positionTensor,
                                              normalizeTensor)  # normalize points
            agentPathTensor = tf.concat([agentPathTensor, [positionTensor]],
                                        axis=0)

        sizeAgentPath = agentPathTensor.get_shape().as_list()
        inputTensor = tf.zeros([0, 7, 5])
        timestampTensor = tf.zeros([0])
        for s in range(sizeAgentPath[0] - 7):
            oneInputTensor = agentPathTensor[s:s + 7, :]
            timestamp = oneInputTensor[6][0]
            timestampTensor = tf.concat([timestampTensor, [timestamp]], axis=0)
            inputTensor = tf.concat([inputTensor, [oneInputTensor]], axis=0)

        sizeInputtensor = inputTensor.get_shape().as_list()

        if sizeInputtensor[0] != 0:
            prediction = network.predict(inputTensor)
            for a in range(len(prediction)):
                pathDict = {"timeStampStart": (timeArray[a+7]),
                            "probability": 1, "steps": []}
                for b in range(len(prediction[a])):
                    timegap = timeArray[a] - timeArray[a - 1]
                    stepDict = {
                        "timestamp": (timeArray[a] + (timegap * b)),
                        "x": prediction[a][b][0] * unNormalizeX.item(),
                        "y": prediction[a][b][1] * unNormalizeY.item()}

                    pathDict["steps"].append(stepDict)
                agentDict["predicted_paths"].append(pathDict)
        predictionDict["agents"].append(agentDict)
    predictionData["predictionData"].append(predictionDict)
    write_json(predictionData, "output")


def main():
    mypath = "StoredData"  # create folder if not present
    if not os.path.isdir(mypath):
        os.makedirs(mypath)

    predicting()


if __name__ == '__main__':
    main()
