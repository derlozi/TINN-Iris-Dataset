# TINN-Iris-Dataset
This is a small project in which I want to test the TINN library with the Iris dataset.

<h3>How does it work?</h3>

The programm reads the dataset out of the text file and trains the net once with the whole set. It then computes the average error over the whole training session and compares it to a certain minimum. If it is reached, the training is stopped. Then, the Net predicts some test cases that were not used in the trainig. It's success rate is printed out, along with any failed attempts. If you want to play around with some values, you can experiment with the learning rate and the target error.
