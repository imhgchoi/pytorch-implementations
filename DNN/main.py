from DNN.data import Data
from DNN.dnn import DNN
from DNN.learner import Learner


if __name__ == '__main__':
    DATA_DIR = 'D:/rawDataFiles/digit_train.csv'

    LEARNING_RATE = 1e-4
    EPOCHS = 200
    BATCH_SIZE = 3000


    data = Data(DATA_DIR)
    target, input = data.import_data()
    dnn = DNN(LEARNING_RATE)
    target = target.to(dnn.device)
    input = input.to(dnn.device)
    print(target.shape)
    print(input.shape)

    learner = Learner(dnn, input, target, batch_size=BATCH_SIZE, epochs=EPOCHS)
    learner.learn()
