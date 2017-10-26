# Machine Learning by the Neural Network

Some basic classes implementing the staddard machine learning.

- ImageLabelSet.java

Abstracted Image &amp; Label set. It includess BasicImage &amp; BaseLabel class
which are the base abstract classes for image and label.

BaseImage.getWidth() must return the width of the image or the number of nodes
contained in the neural network output. BaseImage.height() must return the height
of the image or return 1 for the neural network output.

- QuadLossFunction.java

This class implements ILossFunction which provides the loss function calculation.
the quadratic loss function derives the lenear loss derivative.

- SimpleNet.java

The simple perceptron implementation. The forward() is the forward operation.
calc_deriv_b() and calc_deriv_w() implement the Inverse error propagation method.

Calculation must be ordered: you first call forward(), then calc_deriv_b().
You must call calc_deriv_w() in the last.

- NeuralNet.java

This class represents the generalized Neural Network. The deep neural network 
can be constructed by addeing hidden node numbers.

The forward() cascades perceptrons by connecting perptron's output to the next level perceptron's input.
The backPropagate() implements the back propagation method.

The backPropagate() requires 3 parameters: ImageLabelSet, nset, batchsize.
Batchsize is the child set size from the ImageLabelSet.
The child set will be picked up by the plain random generator from \[0, ImageLabelSet.getQuantity()\).
Nset is the count to repeat the training for the batch.

You must choice the nset carefully; it must be large for predictability, and must be small to prevent
the over-training. In experiment I feel better to set nset to 10.

- AutoEncoder comments are provided at the autoencoder directory.
