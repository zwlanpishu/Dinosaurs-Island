import numpy as np 
from utils import *
import random
from random import shuffle

data = open("dinos.txt", 'r').read()
data = data.lower()
chars = list(set(data))

data_size, vocab_size = len(data), len(chars)
print("There are %d total characters and %d unique characters in your data" %(data_size, vocab_size))

# build dict(map) of the char to index and index to char
char_to_ix = {ch : i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i : ch for i, ch in enumerate(sorted(chars))}


def clip_gradient(gradients, maxValue) : 
    """
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    """
    
    [dWaa, dWax, dWya, db, dby] = (gradients["dWaa"], gradients["dWax"], gradients["dWya"], 
                                   gradients["db"], gradients["dby"])
    
    for gradient in [dWaa, dWax, dWya, db, dby] : 
        np.clip(gradient, -maxValue, maxValue, gradient)

    gradients = {"dWaa" : dWaa, "dWax" : dWax, "dWya" : dWya, "db" : db, "dby" : dby}
    return gradients


def sample(parameters, char_to_ix, seed) : 
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """

    # retrieve parameters and relevant shapes
    Waa, Wax, Wya, by, b = (parameters["Waa"], parameters["Wax"], parameters["Wya"], 
                            parameters["by"], parameters["b"])
    vocab_size = by.shape[0]
    dim_a = Waa.shape[1]

    # create the one-hot vector for the first character   
    x_curr = np.zeros((vocab_size, 1))
    a_prev = np.zeros((dim_a, 1))

    # this is a list containing all the generated indices of the characters
    indices = []
    index = -1

    counter = 0
    newline_character = char_to_ix["\n"]

    # loop to sample until encounter the "\n" or limitation
    while (index != newline_character and counter != 50) : 
        
        a_curr = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x_curr) + b)
        y_curr = softmax(np.dot(Wya, a_curr) + by)

        np.random.seed(counter + seed)
        index = np.random.choice(list(range(vocab_size)), p = y_curr.ravel())
        indices.append(index)

        a_prev = a_curr
        x_curr = np.zeros((vocab_size, 1))
        x_curr[index] = 1

        seed += 1
        counter += 1
    
    if counter == 50 :
        indices.append(newline_character)
    
    return indices


def optimize(X, Y, a_prev, parameters, learning_rate = 0.002) :
    """
    Execute one step of the optimization to train the model
    Argument:
        X -- list of integers, each represent a character
        Y -- list of integers, each represent a character(X shifted on index by left)
        a_prev -- previous hidden state

    Return:
        loss -- value of loss fuction
        gradients -- gradients of parameters
    """ 

    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip_gradient(gradients, 5)
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]


def model(data, ix_to_char, char_to_ix, num_iterations = 60000, 
          n_a = 50, dino_names = 7, vocab_size = 27) : 
    """
    Trains the model and generates dinosaur names. 
    
    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text, size of the vocabulary
    
    Returns:
    parameters -- learned parameters
    """

    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)

    # Initialize loss (this is required because we want to smooth our loss, don't worry about it)
    loss = get_initial_loss(vocab_size, dino_names)

    # Build list of all dinosaur names (training examples)
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)

    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))

    for j in range(num_iterations) : 
        
        # Use the hint above to define one training example (X,Y) (≈ 2 lines)
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]] 
        Y = X[1:] + [char_to_ix["\n"]]

        # Perform one optimization step
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        if j % 2000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            
            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names) : 
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                seed += 1
            print("\n")

    return parameters

parameters = model(data, ix_to_char, char_to_ix)












