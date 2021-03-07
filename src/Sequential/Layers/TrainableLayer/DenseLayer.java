package Sequential.Layers.TrainableLayer;

import Sequential.WeightInitialization.*;
import Sequential.Util.*;
import Sequential.SequentialExceptions.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

import static java.lang.String.format;

public class DenseLayer extends Trainable {
    private int numNodes;
    private int prevSize;
    private HeNormal wInit;
    private double[][] adjustWeights;
    private double[][] prevFirstMoments;
    private double[][] prevSecondMoments;
    private double[][] weights;

    private int numWAdjustments = 0;
    private double totalWAdjustment = 0;
    private int numBAdjustments = 0;
    private double totalBAdjustment = 0;

    public DenseLayer(int numNodes, int[] inDim)
            throws InvalidDimensionException, InvalidOperationException{
        super(inDim);
        this.numNodes = numNodes;
        prevSize = inDim[0] * inDim[1] * inDim[2];
        wInit = new HeNormal();
        wInit.setIn(prevSize);
        weights = new double[prevSize + 1][numNodes];


        // Initialize layer weights
        // The values of the linear jacobian matrix for the layer is the weights
        for (int i = 0; i < weights.length; i++){
            for (int j = 0; j < weights[0].length; j++){
                if (i != weights.length - 1){
                    weights[i][j] = wInit.initializeWeight();
                }
                // Biases are in the last row, they should be init to 0
                else {
                    weights[i][j] = 0;
                }

            }
        }

        // Initialize prevFirstMoments, and prevSecondMoments to 0
        prevFirstMoments = new double[weights.length][weights[0].length];
        NetUtil.fill2D(prevFirstMoments, 0);
        prevSecondMoments = new double[weights.length][weights[0].length];
        NetUtil.fill2D(prevSecondMoments, 0);
    }

    // Add a row to the given two dim array to represent the bias or intercept
    private double[][] addBiasCol(double[][] in){
        double[][] ret = new double[in.length][in[0].length + 1];
        // Copy
        for (int i = 0; i < in.length; i++){
            System.arraycopy(in[i], 0, ret[i], 0, in[i].length);
        }
        // Add bias
        for (int i = 0; i < in.length; i++){
            ret[i][ret[0].length - 1] = 1;
        }
        return ret;
    }

    // Remove the bias row from the given array (Remove last row)
    public double[][] removeBiasRow(double[][] in){
        double[][] ret = new double[in.length - 1][in[0].length];
        for (int i = 0; i < ret.length; i++){
            System.arraycopy(in[i], 0, ret[i], 0, ret[i].length);
        }
        return ret;
    }

    // Performs a forward pass through the layer with only one input
    @Override
    public double[] forwardPass(double[] in) throws InvalidDimensionException{
        double[][] twoDIn = new double[1][in.length + 1];

        System.arraycopy(in, 0, twoDIn[0], 0, in.length);
        twoDIn[0][twoDIn.length] = 1;

        double[][] ret = NetUtil.matMult(twoDIn, weights);
        return NetUtil.two2OneD(ret);
    }

    // Sends a batch of input through the layer
    public double[][] forwardBatchPass(double[][] in) throws InvalidDimensionException{
        // Copy input array into a larger arraay to add a column of biases. Don't fill last col yet
        double[][] formedIn = addBiasCol(in);
        layerResult = NetUtil.matMult(formedIn, weights);
        return layerResult;
    }

    // Compute the gradients of this layer's input with respect to the gradients of the
    // loss to this layers output, and compute the gradients of this layers weights with
    // respect to the gradients of the loss with respect to this layer's output
    public double[][] computeGradients(double[][] jacob, double[][] prevInput) throws InvalidDimensionException{
        if (jacob.length != prevInput.length){
            throw new InvalidDimensionException("Dense: ComputeGradients: Mismatch in batch size with jacobian and given input");
        }
        double[][] ret = NetUtil.matMult(jacob, NetUtil.transpose(removeBiasRow(weights)));
        double[][] formedIn = addBiasCol(prevInput);
        adjustWeights = NetUtil.matMult(NetUtil.transpose(formedIn), jacob);
        return ret;
    }

    // Given the gradients of this layers weights, make appropriate adjustments
    public void updateParamsMiniBatch(double learnRate) throws InvalidDimensionException{
        for (int i = 0; i < adjustWeights.length; i++){
            for (int j = 0; j < adjustWeights[0].length; j++){
                adjustWeights[i][j] *= -1 * learnRate;
            }
        }
        // Perform element weise addition on the weight matrix
        NetUtil.elAddInc(weights, adjustWeights);
    }

    // Update hte parameters for htis layer using an adam approach
    public void updateParamsAdam(double alpha, double b1, double b2, double e){
        double[][] firstMoments = new double[weights.length][weights[0].length];
        double[][] secondMoments = new double[weights.length][weights[0].length];

        double adjA = alpha * Math.sqrt(1 - b2)/(1 - b1);
        for (int i = 0; i < firstMoments.length; i++){
            for (int j = 0; j < firstMoments[0].length; j++){
                firstMoments[i][j] = b1 * prevFirstMoments[i][j] + (1 - b1) * adjustWeights[i][j];
                secondMoments[i][j] = b2 * prevSecondMoments[i][j] + (1 - b2) * Math.pow(adjustWeights[i][j], 2);
                weights[i][j] -= adjA * firstMoments[i][j] / Math.sqrt(secondMoments[i][j] + e);
            }
        }
        prevFirstMoments = firstMoments;
        prevSecondMoments = secondMoments;
    }

    // Returns the output dimensions of the layer
    @Override
    public int[] getOutDim(){
        int[] outDim = {1, 1, numNodes};
        return outDim;
    }

    // Print the info for this layer
    @Override
    public void printInfo(){
        System.out.println(format("Dense Layer: %4d nodes. In: " + Arrays.toString(inDim) + " Out: " + Arrays.toString(getOutDim()), numNodes));
    }

    // Print the trining info for this layer
    @Override
    public void printTrainInfo(){
        System.out.println("Dense Layer. Num weight adjustments = " + numWAdjustments + ". Average weight adjustment = " + totalWAdjustment / numWAdjustments +
                " Num bias adjustments = " + numBAdjustments + ". Average bias adjustment = " + totalBAdjustment / numBAdjustments);
    }

    // Given a scanner to a file, fill the parameters of this layer
    public void fillParams(Scanner scIn){
        for (int i = 0; i < weights.length; i++){
            for (int j = 0; j < weights[0].length; j++){
                weights[i][j] = scIn.nextDouble();
            }
        }
        // Git rid of line buffer
        scIn.nextLine();
    }

    // Given a filewriter, write the layer by writing a metaline
    // containing -n for number of nodes, -a for activation function,
    // -d for pool dimensions, and -s for stride length. Precede these
    // options by writing a d for dense layer
    @Override
    public void writeLayer(FileWriter fWrite) throws IOException {
        StringBuilder outString = new StringBuilder();
        outString.append("dense -n ");
        outString.append(numNodes);
        outString.append("\n");
        // output the parameters of the layer
        for (int i = 0; i < weights.length; i++){
            for (int j = 0; j < weights[0].length; j++){
                outString.append(weights[i][j]);
                outString.append(" ");
            }
            outString.append("\n");
        }
        fWrite.write(outString.toString());
    }
}
