// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential;

import Sequential.Layers.TrainableLayer.*;
import Sequential.Layers.*;
import Sequential.LossFunctions.*;
import Sequential.Optimizers.*;
import Sequential.Optimizers.Optimizer;
import Sequential.SequentialExceptions.*;
import Sequential.Util.*;
import static java.lang.String.format;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Class representing a neural network.
 */
public class Sequential {
    private ArrayList<Layer> layers;
    private Optimizer opt = new Adam();
    private LossFunction lossFunc = new MeanSquaredError();
    private String[] metrics;
    private int[] inDim;

    /**
     * Constructs a new Sequential object.
     */
    public Sequential(){
        layers = new ArrayList<Layer>(0);
    }

    /**
     * Sets the loss function, optimizer, and metrics of the network to their default value.
     * That is, the Sequential object will use Mean Squared Error for loss, Adam for its optimizer, and
     * will not provide any metrics other than loss.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    public void compile() throws InvalidOperationException, InvalidDimensionException{
        this.metrics = new String[0];
        // If previous layer is a tranformation layer, we need to add a default activation function
        // Maxpools don't need activation
        if (getLastLayer() instanceof Trainable){
            addSoftMax();
        }
    }

    /**
     * Sets the loss function of the network to the given loss function, and sets the optimizer and metrics to their default values.
     * That is, the Sequential object will use Adam for its optimizer, and will not provide any metrics other than loss.
     * @param lossFunc The loss function to be used by the Sequential Object.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    public void compile(LossFunction lossFunc) throws InvalidOperationException, InvalidDimensionException {
        compile();
        this.lossFunc = lossFunc;

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
    public void compile(LossFunction lossFunc, String[] metrics) throws InvalidOperationException, InvalidDimensionException {
        compile(lossFunc);
        this.metrics = metrics;
    }

    /**
     * Sets the loss function, optimizer, and metrics of the network to the given loss function, optimizer, and metrics.
     * @param lossFunc The loss function to be used by the network.
     * @param opt The optimizer to be used by the network.
     * @param metrics The metrics to be used by the network.
     * @throws InvalidOperationException
     * @throws InvalidDimensionException
     */
    public void compile(LossFunction lossFunc, Optimizer opt, String[] metrics) throws InvalidOperationException, InvalidDimensionException {
        compile(lossFunc, metrics);
        this.opt = opt;
    }

    // Return the partial derivatives of the loss function with respect to the
    // activations of the final layer
    public double[][] getLossJacobian(double[][] expected) throws InvalidDimensionException{
        double[][] lastAct = layers.get(layers.size() - 1).getLayerResult();
        if (lastAct.length != expected.length || lastAct[0].length != expected[0].length){
            throw new InvalidDimensionException("calculateBatchLoss: mismatch dimensions of expected vs actual");
        }
        double[][] ret = new double[lastAct.length][lastAct[0].length];
        for (int i = 0; i < lastAct.length; i++){
            ret[i] = lossFunc.calculatePDerivatives(expected[i], lastAct[i]);
            // derivatives need to be multiplied by 1/batchsize because loss is calculated by doing this
            for (int j = 0; j < lastAct[0].length; j++){
                ret[i][j] = ret[i][j] / lastAct.length;
            }
        }
        return ret;
    }

    // Perform a prediction by making a forward pass through the network.
    public void predict(double[][][][] input) throws InvalidDimensionException{
        double[] flatIn;

        for (int i = 0; i < input.length; i++){
            flatIn = NetUtil.three2OneD(input[i]);
            for (int j = 0; j < layers.size(); j++){
                flatIn = layers.get(i).forwardPass(flatIn);
            }
            System.out.println("Prediction: " + Arrays.toString(flatIn));
        }
    }

    // Perform a foward pass of a batch through the network
    public double[][] forwardBatchPass(double[][] input) throws InvalidDimensionException{
        // Perform pass
        for (int i = 0; i < layers.size(); i++){
            input = layers.get(i).forwardBatchPass(input);
        }
        return input;
    }

    // Perform a forward pass (not batch-wise) through the network
    private double[] forwardPass(double[] input) throws InvalidDimensionException{
        for (int i = 0; i < layers.size(); i++){
            input = layers.get(i).forwardPass(input);
        }
        return input;
    }

    // Return the layers arraylist for the network
    public ArrayList<Layer> getLayers(){
        return layers;
    }

    // Return the last layer of the network
    public Layer getLastLayer(){
        return layers.get(layers.size() - 1);
    }

    // Return the loss function for the network
    public LossFunction getLossFunc() {
        return lossFunc;
    }

    // Return the optimizer for the network
    public Optimizer getOptimizer(){
        return opt;
    }

    // TODO - CHECK THAT THIS IS A VALID OPERATION
    public void fit(double[][][][] x, double[][] y, int batchSize, int epochs)
            throws InvalidOperationException, InvalidDimensionException{
        if (this.metrics == null){
            throw new InvalidOperationException("Must compile the network before trying to train the network");
        }
        opt.constructOptimizer(this);
        opt.trainNetwork(x, y, batchSize, epochs);
    }


    // TODO -- I COULD OPTIMIZE THIS
    // Evaluates the network by running each testing pair individually and not in
    // batches
    public void evaluate(double[][][][] x, double[][] y) throws InvalidOperationException,
            InvalidDimensionException{
        if (metrics == null){
            throw new InvalidOperationException("Must compile the network before evaluating the network");
        }
        if (x.length != y.length){
            throw new InvalidDimensionException("Must have same number of input examples as output examples");
        }
        System.out.print("Evaluation: ");
        boolean calcAccuracy = false;
        double[][] flatBatch = new double[x.length][x[0].length * x[1].length * x[2].length];
        MetricCalculator ms = new MetricCalculator(getLastLayer(), lossFunc);
        double totalLoss = 0;
        double numCorrect = 0;
        double[] result;

        // Flatten the batches
        for (int i = 0; i < x.length; i++){
            flatBatch[i] = NetUtil.three2OneD(x[i]);
        }

        // See what metrics we need to calculate
        for (int i = 0; i < metrics.length; i++){
            if (metrics[i].equalsIgnoreCase("accuracy")){
                calcAccuracy = true;
                break;
            }
        }

        // Calculate stats for each training example
        for (int i = 0; i < x.length; i++){
            result = forwardPass(NetUtil.three2OneD(x[i]));
            totalLoss += lossFunc.calculateLoss(y[i], result);
            if (calcAccuracy){
                numCorrect += ms.isCorrect(result, y[i]);
            }
        }

        System.out.print(format("Loss: %5f", totalLoss / x.length));
        if (calcAccuracy){
            System.out.print(format(" Accuracy: %5f", numCorrect / x.length));
        }
        System.out.println();
    }

    // Prints information about the network
    public void printInfo(){
        for (int i = 0; i < layers.size(); i++){
            layers.get(i).printInfo();
        }
    }

    // Return the input dimensions for the network
    public int[] getInDim(){
        return inDim;
    }

    // Return the metrics for this network
    public String[] getMetrics(){
        return metrics;
    }

    // Adds a dense layer to the network with the given activation function
    // and number of nodes. If the input parameter is null, pass the output
    // dimension of the previous layer
    public void addDense(int numNodes, int[] inDim, String actFunc)
            throws InvalidOperationException, InvalidDimensionException{
        if (inDim == null){
            if (layers.size() == 0){
                throw new InvalidOperationException("First layer must be given an initial size");
            }
            inDim = layers.get(layers.size() - 1).getOutDim();
        }
        else {
            this.inDim = inDim;
        }

        // If previous layer is a tranformation layer, we need to add a default activation function
        // Maxpools don't need activation
        if (layers.size() > 0 && getLastLayer() instanceof Trainable){
            addRelu();
        }

        layers.add(new DenseLayer(numNodes, inDim));
        // Optionally add activation function after this layer
        if (actFunc != null){
            layers.add(ActivationDisbatch.getActFunc(actFunc, layers.get(layers.size() - 1).getOutDim()));
        }
    }

    // Adds a convolutional layer to the network.
    // If one of the given optional parameters is null, set it to a
    // default value before initializing the layer
    public void addConv(Integer numFilters, int[] filterDims, int[] inDim, int[] strideLength, String actFunc)
            throws InvalidOperationException, InvalidDimensionException{
        if (inDim == null){
            if (layers.size() == 0){
                throw new InvalidOperationException("First layer must be given an initial size");
            }
            inDim = layers.get(layers.size() - 1).getOutDim();
        } else{
            this.inDim = inDim;
        }

        // If previous layer is a tranformation layer, we need to add a default activation function
        // Maxpools don't need activation
        if (layers.size() > 0 && getLastLayer() instanceof Trainable){
            addRelu();
        }

        // Set default stride length
        if (strideLength == null){
            strideLength = new int[2];
            strideLength[0] = 1;
            strideLength[1] = 1;
        }
        layers.add(new ConvLayer(numFilters, filterDims, inDim, strideLength));
        // optinally add an activation layer
        if (actFunc != null){
            layers.add(ActivationDisbatch.getActFunc(actFunc, layers.get(layers.size() - 1).getOutDim()));
        }
    }

    // Adds a maxpool layer to the network.
    // If one of the given optional parameters is null, set it to a
    // default value before initializing the layer
    public void addMaxPool(int[] poolDims, int[] inDim, int[] strideLength)
            throws InvalidOperationException, InvalidDimensionException{
        if (inDim == null){
            if (layers.size() == 0){
                throw new InvalidOperationException("First layer must be given an initial size");
            }
            inDim = layers.get(layers.size() - 1).getOutDim();
        } else {
            this.inDim = inDim;
        }

        // If previous layer is a tranformation layer, we need to add a default activation function
        // Maxpools don't need activation
        if (layers.size() > 0 && getLastLayer() instanceof Trainable){
            addRelu();
        }

        if (strideLength == null){
            strideLength = new int[2];
            strideLength[0] = poolDims[0];
            strideLength[1] = poolDims[1];
        }
        layers.add(new MaxPool(poolDims, inDim, strideLength));
    }

    // Adds a Relu activation layer to the network
    public void addRelu() throws InvalidOperationException, InvalidDimensionException{
        if (layers.size() == 0){
            throw new InvalidOperationException("First network layer cannot be an activation layer");
        }
        layers.add(new ReluLayer(getLastLayer().getOutDim()));
    }

    // Adds a Relu activation layer to the network
    public void addSigmoid() throws InvalidOperationException, InvalidDimensionException{
        if (layers.size() == 0){
            throw new InvalidOperationException("First network layer cannot be an activation layer");
        }
        layers.add(new SigmoidLayer(getLastLayer().getOutDim()));
    }

    // Adds a Relu activation layer to the network
    public void addSoftMax() throws InvalidOperationException, InvalidDimensionException{
        if (layers.size() == 0){
            throw new InvalidOperationException("First network layer cannot be an activation layer");
        }
        layers.add(new SoftMaxLayer(getLastLayer().getOutDim()));
    }
}
