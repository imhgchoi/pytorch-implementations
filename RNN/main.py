from RNN.data import Data
from RNN.rnn import RNN
from RNN.learner import Learner


if __name__ == '__main__':
    DATA_DIR = 'D:/rawDataFiles/stock_data/AAPL.csv'

    LEARNING_RATE = 0.00003
    EPOCHS = 100
    BATCH_SIZE = 2000


    data = Data(DATA_DIR)
    input, target = data.import_data()
    rnn = RNN(LEARNING_RATE)
    target = target.to(rnn.device)
    input = input.to(rnn.device)
    print(target.shape)
    print(input.shape)

    learner = Learner(rnn, input, target, data.time_step, batch_size=BATCH_SIZE, epochs=EPOCHS)
    learner.learn()
