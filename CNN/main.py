from CNN.data import Data
from CNN.cnn import CNN
from CNN.learner import Learner


if __name__ == '__main__':
    DATA_DIR = 'D:/rawDataFiles/digit_train.csv'

    LEARNING_RATE = 5e-5
    EPOCHS = 100
    BATCH_SIZE = 3000


    data = Data(DATA_DIR)
    target, input = data.import_data()
    cnn = CNN(LEARNING_RATE)
    target = target.to(cnn.device)
    input = input.to(cnn.device)
    print(target.shape)
    print(input.shape)

    learner = Learner(cnn, input, target, batch_size=BATCH_SIZE, epochs=EPOCHS)
    learner.learn()
