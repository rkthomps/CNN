package UI;

import Sequential.LossFunctions.LossFunction;
import Sequential.NetOps.NetworkLoader;
import Sequential.NetOps.NetworkWriter;
import Sequential.Optimizers.Optimizer;
import Sequential.Sequential;
import Sequential.SequentialExceptions.InvalidDimensionException;
import Sequential.SequentialExceptions.InvalidNetworkFormatException;
import Sequential.SequentialExceptions.InvalidOperationException;

import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * User interface to build convolutional neural networks.
 */
public class NeuralNetwork {
    Sequential net;
    NetworkWriter netWrite;
    NetworkLoader netLoad;

    /**
     * Construct a UI.NeuralNetwork object.
     */
    public NeuralNetwork(){
        this.net = new Sequential();
        this.netWrite = new NetworkWriter(net);
        this.netLoad = new NetworkLoader();
    }

    /**
     * Sets the loss function, optimizer, and metrics of the network to their default value.
     * That is, the Sequential object will use Mean Squared Error for loss, Adam for its optimizer, and
     * will not provide any metrics other than loss.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    public void compile() throws InvalidOperationException, InvalidDimensionException{
        net.compile();
    }

    /**
     * Sets the loss function of the network to the given loss function, and sets the optimizer and metrics to their default values.
     * That is, the Sequential object will use Adam for its optimizer, and will not provide any metrics other than loss.
     * @param lossFunc The loss function to be used by the Sequential Object.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    public void compile(LossFunction lossFunc) throws InvalidOperationException, InvalidDimensionException{
        net.compile(lossFunc);
    }

    /**
     * Sets the loss function and metrics of the network to the given loss function and metrics. Sets the optimizer to its default value.
     * That is, the Sequential object will use Adam for its optimizer.
     * @param lossFunc The Loss function to be used by the Sequential Object
     * @param metrics An array of metrics that the Sequential object should track during training and evaluation.
     *                Options include: accuracy.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    public void compile(LossFunction lossFunc, String[] metrics) throws InvalidOperationException, InvalidDimensionException{
        net.compile(lossFunc, metrics);
    }

    /**
     * Sets the loss function, optimizer, and metrics of the network to the given loss function, optimizer, and metrics.
     * @param lossFunc The loss function to be used by the network.
     * @param opt The optimizer to be used by the network.
     * @param metrics The metrics to be used by the network.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    public void compile(LossFunction lossFunc, Optimizer opt, String[] metrics) throws InvalidOperationException, InvalidDimensionException{
        net.compile(lossFunc, opt, metrics);
    }

    /**
     * Prints predictions for an array of inputs.
     * @param inputs Array of 3D inputs to the network.
     */
    public void predict (double[][][][] inputs) throws InvalidDimensionException{
        net.predict(inputs);
    }


    /**
     * Fit the network to an approximate mapping between the given inputs and outputs.
     * @param x Array of 3D inputs to the network for training.
     * @param y Array of 1D expected outputs corresponding to the inputs.
     * @param batchsize Number of inputs that should be in training batches.
     * @param epochs Number of times the inputs should be trained on.
     * @throws InvalidDimensionException
     * @throws InvalidOperationException
     */
    public void fit (double[][][][] x, double[][] y, int batchsize, int epochs)
            throws InvalidDimensionException, InvalidOperationException{
        net.fit(x, y, batchsize, epochs);
    }


    /**
     * Provide performance metrics measuring the networks ability to map the given inputs to the given outputs.
     * @param x Array of 3D inputs to the network to evaluate.
     * @param y Array of 1D expected outputs corresponding to the inputs
     * @throws InvalidDimensionException
     * @throws InvalidOperationException
     */
    public void evaluate(double[][][][] x, double[][] y)
            throws InvalidDimensionException, InvalidOperationException{
        net.evaluate(x, y);
    }

    /**
     * Provide dimensional information about every layer in the network.
     */
    public void printInfo(){
        net.printInfo();
    }

    /**
     * Serializes the network to a text file for storage.
     * @param path Path to the location where the network should be serialized.
     * @throws IOException
     */
    public void save(String path) throws IOException {
        netWrite.writeNetwork(path);
    }

    /**
     * Reads the serialized form of the network from the file specified by the path.
     * @param path Path to the location of serialized network to be read.
     * @throws FileNotFoundException
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     * @throws InvalidNetworkFormatException
     */
    public void load(String path)
            throws FileNotFoundException, InvalidOperationException, InvalidDimensionException, InvalidNetworkFormatException {
        this.net = netLoad.loadNetwork(path);
        this.netWrite = new NetworkWriter(this.net);
    }

    /**
     * Add a Relu activation layer to the network.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    public void addRelu() throws InvalidOperationException, InvalidDimensionException{
        net.addRelu();
    }

    /**
     * Add a Sigmoid activation layer to the network.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    public void addSigmoid() throws InvalidOperationException, InvalidDimensionException{
        net.addSigmoid();
    }

    /**
     * Add a Softmax activation layer to the network.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    public void addSoftMax() throws InvalidOperationException, InvalidDimensionException{
        net.addSoftMax();
    }

    /**
     * Add a Dense layer to the network.
     * @param numNodes The number of nodes the dense layer should have.
     * @param inputSize Dimensionality of a single training example. Must be three element representing: Depth, height, width.
     * @param actFunc String corresponding to the type of Activation layer that should follow the dense layer.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    // User-friendly add dense function
    public void addDense(int numNodes, int[] inputSize , String actFunc)
            throws InvalidOperationException, InvalidDimensionException {
        net.addDense(numNodes, inputSize, actFunc);
    }

    /**
     * Add a Dense layer to the network.
     * @param numNodes The number of nodes the dense layer should have.
     * @param inputSize Dimensionality of a single training example. Must be three element representing: Depth, height, width.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    public void addDense(int numNodes, int[] inputSize)
            throws InvalidOperationException, InvalidDimensionException {
        net.addDense(numNodes, inputSize, null);
    }


    /**
     * Add a Dense layer to the network.
     * @param numNodes The number of nodes the dense layer should have.
     * @param actFunc String corresponding to the type of Activation layer that should follow the dense layer.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    public void addDense(int numNodes, String actFunc)
            throws InvalidOperationException, InvalidDimensionException {
        net.addDense(numNodes, null, actFunc);
    }

    /**
     * Add a Dense layer to the network.
     * @param numNodes The number of nodes the dense layer should have.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    public void addDense(int numNodes)
            throws InvalidOperationException, InvalidDimensionException {
        net.addDense(numNodes, null, null);
    }

    /**
     * Adds a convolutional layer to the network.
     * @param numFilters Number of filters that should be in the layer.
     * @param filterDims Two element array representing the dimensions of the filter.
     *                 First element is the vertical size; second element is the horizontal size.
     * @param inputSize Dimensionality of a single training example. Must be three element representing: Depth, height, width.
     * @param strideLength Two element array representing the vertical and horizontal stridelengths.
     *                     First element is the vertical stride. Second is the horizontal stride.
     * @param actFunc String corresponding to the type of Activation layer that should follow the convolutional layer.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    // User-friendly add conv function
    public void addConv(int numFilters, int[] filterDims, int[] inputSize, int[] strideLength, String actFunc)
            throws InvalidOperationException, InvalidDimensionException {
        net.addConv(numFilters, filterDims, inputSize, strideLength, actFunc);
    }

    /**
     * Adds a convolutional layer to the network.
     * @param numFilters Number of filters that should be in the layer.
     * @param filterDims Two element array representing the dimensions of the filter.
     *                 First element is the vertical size; second element is the horizontal size.
     * @param inputSize Dimensionality of a single training example. Must be three element representing: Depth, height, width.
     * @param strideLength Two element array representing the vertical and horizontal stridelengths.
     *                     First element is the vertical stride. Second is the horizontal stride.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    // User-friendly add conv function
    public void addConv(int numFilters, int[] filterDims, int[] inputSize, int[] strideLength)
            throws InvalidOperationException, InvalidDimensionException {
        net.addConv(numFilters, filterDims, inputSize, strideLength, null);
    }

    /**
     * Adds a convolutional layer to the network. Assumes vertical and horizontal stride length is one.
     * @param numFilters Number of filters that should be in the layer.
     * @param filterDims Two element array representing the dimensions of the filter.
     *                 First element is the vertical size; second element is the horizontal size.
     * @param inputSize Dimensionality of a single training example. Must be three element representing: Depth, height, width.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    // User-friendly add conv function
    public void addConv(int numFilters, int[] filterDims, int[] inputSize)
            throws InvalidOperationException, InvalidDimensionException {
        net.addConv(numFilters, filterDims, inputSize, null, null);
    }

    /**
     * Adds a convolutional layer to the network. Assumes vertical and horizontal stride length is one.
     * @param numFilters Number of filters that should be in the layer.
     * @param filterDims Two element array representing the dimensions of the filter.
     *                 First element is the vertical size; second element is the horizontal size.
     * @param actFunc String corresponding to the type of Activation layer that should follow the convolutional layer.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    // User-friendly add conv function
    public void addConv(int numFilters, int[] filterDims, int[] inputSize, String actFunc)
            throws InvalidOperationException, InvalidDimensionException {
        net.addConv(numFilters, filterDims, inputSize, null, actFunc);
    }

    /**
     * Adds a convolutional layer to the network.
     * @param numFilters Number of filters that should be in the layer.
     * @param filterDims Two element array representing the dimensions of the filter.
     *                 First element is the vertical size; second element is the horizontal size.
     * @param actFunc String corresponding to the type of Activation layer that should follow the convolutional layer.
     * @param strideLength Two element array representing the vertical and horizontal stridelengths.
     *                     First element is the vertical stride. Second is the horizontal stride.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    // User-friendly add conv function
    public void addConv(int numFilters, int[] filterDims, String actFunc, int[] strideLength)
            throws InvalidOperationException, InvalidDimensionException {
        net.addConv(numFilters, filterDims, null, strideLength, actFunc);
    }

    /**
     * Adds a convolutional layer to the network. Assumes vertical and horizontal stride length is one.
     * @param numFilters Number of filters that should be in the layer.
     * @param filterDims Two element array representing the dimensions of the filter.
     *                 First element is the vertical size; second element is the horizontal size.
     * @param actFunc String corresponding to the type of Activation layer that should follow the convolutional layer.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    // User-friendly add conv function
    public void addConv(int numFilters, int[] filterDims, String actFunc)
            throws InvalidOperationException, InvalidDimensionException {
        net.addConv(numFilters, filterDims, null, null, actFunc);
    }

    /**
     * Adds a convolutional layer to the network. Assumes vertical and horizontal stride length is one.
     * @param numFilters Number of filters that should be in the layer.
     * @param filterDims Two element array representing the dimensions of the filter.
     *                 First element is the vertical size; second element is the horizontal size.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    // User-friendly add conv function
    public void addConv(int numFilters, int[] filterDims)
            throws InvalidOperationException, InvalidDimensionException {
        net.addConv(numFilters, filterDims, null, null, null);
    }

    /**
     * Adds a Max Pool layer to the network.
     * @param poolDims Two element array representing the dimensions of the pool.
     *                 First element is the vertical size; second element is the horizontal size.
     * @param inputSize Dimensionality of a single training example. Must be three element representing: Depth, height, width.
     * @param strideLength Two element array representing the vertical and horizontal stridelengths.
     *                     First element is the vertical stride. Second is the horizontal stride.
     * @throws InvalidDimensionException
     * @throws InvalidOperationException
     */
    public void addMaxPool(int[] poolDims, int[] inputSize, int[] strideLength)
            throws InvalidDimensionException, InvalidOperationException{
        net.addMaxPool(poolDims, inputSize, strideLength);
    }

    /**
     * Adds a Max Pool layer to the network.
     * @param poolDims Two element array representing the dimensions of the pool.
     *                 First element is the vertical size; second element is the horizontal size.
     * @param strideLength Two element array representing the vertical and horizontal stridelengths.
     *                     First element is the vertical stride. Second is the horizontal stride.
     * @throws InvalidDimensionException
     * @throws InvalidOperationException
     */
    public void addMaxPool(int[] poolDims, int[] strideLength)
            throws InvalidDimensionException, InvalidOperationException{
        net.addMaxPool(poolDims, null, strideLength);
    }

    /**
     * Adds a Max Pool layer to the network. Assumes stride lengths equal to the vertical and horizontal sizes of the pool.
     * @param poolDims Two element array representing the dimensions of the pool.
     *                 First element is the vertical size; second element is the horizontal size.
     * @throws InvalidDimensionException
     * @throws InvalidOperationException
     */
    public void addMaxPool(int[] poolDims)
            throws InvalidDimensionException, InvalidOperationException{
        net.addMaxPool(poolDims, null, null);
    }
}
