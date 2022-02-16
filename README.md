# CNN From Scratch

Implemented a convolutional neural network for an image
classification task. There are six basic components in the neural network:

- **Convolution layer :** there will be four (hyper)parameters: the number of output channels,
  filter dimension, stride, padding.
- **Activation layer :** implement an element-wise ReLU.
- **Max-pooling layer :** there will be two parameters: filter dimension, stride.
- **Fully-connected layer :** a dense layer. There will be one parameter: output dimension.
- **Flattening layer :** it will convert a (series of) convolutional filter maps to a column vector.
- **Softmax layer :** it will convert final layer projections to normalized probabilities.

The model architecture should be given in a text file. A sample architecture is given in the input file for convenience. Two models were trained on the given architecture  on MNIST and CIFAR-10 dataset both for 5 epochs and the evaluation scores are given in the files mnist_report.txt and CIFAR-10_report.txt.

## How to run

```bash
$ python cnn.py <input_architecture_file>
```

