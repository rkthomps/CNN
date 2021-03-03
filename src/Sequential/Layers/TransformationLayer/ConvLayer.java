package Sequential.Layers.TransformationLayer;

import Sequential.Layers.Layer;
import Sequential.Optimizers.Adjustor;
import Sequential.WeightInitialization.*;
import Sequential.Util.*;
import Sequential.SequentialExceptions.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

import static java.lang.String.format;

public class ConvLayer extends TransformationLayer {
    // Dimensions of filters are: numFilters, depth, height, width
    private double[][] filters;
    private double[][] prevFirstMoments;
    private double[][] prevSecondMoments;
    private double[][] adjustFilters;
    private HeNormal wInit;
    private int vertStride;
    private int horStride;
    private int[] outDim;
    private int[] filterDim;
    private int[][][] inMap;

    private int numFAdj = 0;
    private double totFAdj = 0;
    private int numBAdj = 0;
    private double totBAdj = 0;

    // Initialize the paramenters of the Sequential.Sequential.Layers.Layer. Stride length overridden
    public ConvLayer(int numFilters, int[] filterDim, int[] inDim, int[] strideLength)
            throws InvalidDimensionException, InvalidOperationException{
        super(inDim);

        if (strideLength.length != 2){
            throw new InvalidDimensionException("Stride length array must be of length 2");
        }
        if (filterDim.length != 2){
            throw new InvalidDimensionException("Spacial window dimension array must be of length 2");
        }
        if (filterDim[0] > inDim[1] || filterDim[1] > inDim[2]){
            throw new InvalidDimensionException("Pool dimensions cannot be greater than layer input dimensions");
        }

        this.filterDim = Arrays.copyOf(filterDim, filterDim.length);
        filters = new double[inDim[0] * filterDim[0] * filterDim[1] + 1][numFilters];
        vertStride = strideLength[0];
        horStride = strideLength[1];
        outDim = new int[3];
        setOutDim();
        wInit = new HeNormal();
        wInit.setIn(outDim[1] * outDim[2]);
        inMap = new int[inDim[0] * inDim[1] * inDim[2]][(filterDim[0] / vertStride + 1) * (filterDim[1] / horStride + 1)][2];
        fillInMap();

        // Initialize filter weights
        for (int i = 0; i < filters.length; i++) {
            for (int j = 0; j < filters[0].length; j++) {
                if (i != filters.length - 1){
                    filters[i][j] = wInit.initializeWeight();
                }
                // Biases are in the last row, they should be  0
                else{
                    filters[i][j] = 0;
                }
            }
        }

        // Initialize past weight adjustments to 0
        prevFirstMoments = new double[filters.length][filters[0].length];
        NetUtil.fill2D(prevFirstMoments, 0);
        prevSecondMoments = new double[filters.length][filters[0].length];
        NetUtil.fill2D(prevSecondMoments, 0);
    }

    // For each index in the flattened 3D input, store in columns the indexes that index is mapped to
    // if the index is mapped to less other indecies than the number of columns, the row will be padded
    // with -1
    private void fillInMap(){
        // provide storage for each row of the inMap to indicate where to add the next location
        int[] curIndex = new int[inMap.length];
        int flatIndex;
        // fill matrix with -1, and the curIndex of each row with 0
        for (int i = 0; i < inMap.length; i++) {
            curIndex[i] = 0;
            for (int j = 0; j < inMap[0].length; j++) {
                for (int k = 0; k < inMap[0][0].length; k++){
                    inMap[i][j][k] = -1;
                }
            }
        }

        // Iterate vertically (moving the filter)
        for (int i = 0; i + filterDim[0] <= inDim[1]; i += vertStride) {
            // Iterate horizontally (moving the filter)
            for (int j = 0; j + filterDim[1] <= inDim[2]; j += horStride) {
                // Iterate through layers of the filter
                for (int l = 0; l < inDim[0]; l++) {
                    // Iterate through rows of the filter
                    for (int r = 0; r < filterDim[0]; r++) {
                        // Iterate through columns of the filter
                        for (int c = 0; c < filterDim[1]; c++) {
                            flatIndex = (l * (inDim[1] * inDim[2]) + (r + i) * inDim[2] + (c + j));
                            inMap[flatIndex][curIndex[flatIndex]][0] = ((i / vertStride) * outDim[2] + (j / horStride));
                            inMap[flatIndex][curIndex[flatIndex]][1] = l * filterDim[0] * filterDim[1] +
                                    r * filterDim[1] + c;
                            curIndex[flatIndex]++;
                        }
                    }
                }
            }
        }
    }

    // Map the values of the input to the appropriate location in the layer's transformed input matrix
    private double[][] formedIn(double[] in) throws InvalidDimensionException{
        if (in.length != inDim[0] * inDim[1] * inDim[2]){
            throw new InvalidDimensionException("Conv Sequential.Sequential.Layers.Layer: fowardPass: Input size doesn't match layer input dim");
        }
        double[][] ret = new double[outDim[1] * outDim[2]][filters.length];
        int j;

        // Assign each input value to its proper location in the formed input
        for (int i = 0; i < in.length; i++){
            j = 0;
            while (inMap[i][j][0] != -1){
                ret[inMap[i][j][0]][inMap[i][j][1]] = in[i];
                j++;
            }
        }

        // Add 1s to the last column of the formed input to account for biases
        for (int i = 0; i < ret.length; i++){
            ret[i][ret[0].length - 1] = 1;
        }
        return ret;
    }

    // Given a flattened input, return a flattened output representing one convolution
    // flattened by column wise flattening. Sequential.Sequential.Layers.Layer outputs are stacked
    public double[] forwardPass(double[] in) throws InvalidDimensionException {
        double[][] formattedIn = formedIn(in);
        double[] ret = NetUtil.two2OneD(NetUtil.matMult(formattedIn, filters));
        return ret;
    }

    // Given a batch of inputs, reuturn a batch of outputs after passing the inputs through the layer
    public double[][] forwardBatchPass(double[][] in) throws InvalidDimensionException {
        layerResult = new double[in.length][outDim[0] * outDim[1] * outDim[2]];
        double[][] formattedIn;

        // Run every input through the input transformation and perform the linear convolution operation
        for (int i = 0; i < in.length; i++){
            formattedIn = formedIn(in[i]);
            layerResult[i] = NetUtil.two2OneD(NetUtil.matMult(formattedIn, filters));
        }

        return layerResult;
    }

    // Compute the gradients of this layer's input with respect to the gradients of the
    // loss to this layers output, and compute the gradients of this layers filters with
    // respect to the gradients of the loss with respect to this layer's output
    public double[][] computeGradients(double[][] jacob, double[][] prevInput) throws InvalidDimensionException{
        if (jacob.length != prevInput.length){
            throw new InvalidDimensionException("Conv: ComputeGradients: Mismatch in batch size with jacobian and given input");
        }
        adjustFilters = new double[filters.length][filters[0].length];
        NetUtil.fill2D(adjustFilters, 0);
        // Each row of above matrix is stacked filters.
        double[][][] jacobSet = NetUtil.two2ThreeD(jacob, jacob.length, outDim[1] * outDim[2], outDim[0]);
        double[][] ret = new double[jacob.length][inDim[0] * inDim[1] * inDim[2]];
        // Aggregate filter gradients with the set of jacobians
        for (int i = 0; i < jacobSet.length; i++){
            NetUtil.elAddInc(adjustFilters, computeFilterGradients(jacobSet[i], prevInput[i]));
            computeInputGradients(ret[i], jacobSet[i]);
        }
        return ret;
    }

    // Given the jacobian of a single batch's output, and the corresponding batch's input,
    // compute the gradients for each of the weights in the filter
    public double[][] computeFilterGradients(double[][] jacobSet, double[] batchIn) throws InvalidDimensionException{
        return NetUtil.matMult(NetUtil.transpose(formedIn(batchIn)), jacobSet);
    }

    // Given the jacobian of a single batch's output, compute the gradients for each of the inputs
    // to the layer and store them in the given ret array
    public void computeInputGradients(double[] ret, double[][] curJacob) throws InvalidDimensionException{
        double[][] formedJacob = NetUtil.matMult(curJacob, NetUtil.transpose(filters));
        int j;
        double gradSum;
        for (int i = 0; i < ret.length; i++){
            j = 0;
            gradSum = 0;
            while (inMap[i][j][0] != -1){
                gradSum += formedJacob[inMap[i][j][0]][inMap[i][j][1]];
                j++;
            }
            ret[i] = gradSum;
        }
    }

    // Update the parameters for this layer using a minibatch approach
    public void updateParamsMiniBatch(double learnRate) throws InvalidDimensionException{
        // First, multiply each value in adjustFilters by the negative learn rate
        for (int i = 0; i < adjustFilters.length; i++){
            for (int j = 0; j < adjustFilters[0].length; j++){
                adjustFilters[i][j] *= -1 * learnRate;
            }
        }
        // Perform elemnt wise addition on the filter matrix
        NetUtil.elMulInc(filters, adjustFilters);
    }

    // Update hte parameters for htis layer using an adam approach
    public void updateParamsAdam(double alpha, double b1, double b2, double e){
        double[][] firstMoments = new double[filters.length][filters[0].length];
        double[][] secondMoments = new double[filters.length][filters[0].length];

        double adjA = alpha * Math.sqrt(1 - b2)/(1 - b1);
        for (int i = 0; i < firstMoments.length; i++){
            for (int j = 0; j < firstMoments[0].length; j++){
                firstMoments[i][j] = b1 * prevFirstMoments[i][j] + (1 - b1) * adjustFilters[i][j];
                secondMoments[i][j] = b2 * prevSecondMoments[i][j] + (1 - b2) * Math.pow(adjustFilters[i][j], 2);
                filters[i][j] -= adjA * firstMoments[i][j] / Math.sqrt(secondMoments[i][j] + e);
            }
        }
        prevFirstMoments = firstMoments;
        prevSecondMoments = secondMoments;
    }

    // Returns the output dimensions of this layer
    @Override
    public int[] getOutDim(){
        return outDim;
    }

    // Print the info for this layer
    @Override
    public void printInfo(){
        System.out.println(format("Conv Layer: %4d filters. Filter Size: " + Arrays.toString(filterDim) +
                " In: " + Arrays.toString(inDim) + " Out: " + Arrays.toString(getOutDim()) + " HorStride: %4d VerStride: %4d",
                filters.length, horStride, vertStride));
    }

    // Print the trining info for this layer
    @Override
    public void printTrainInfo(){
        System.out.println("Conv Layer. Num filter adjustments = " + numFAdj + ". Average filter adjustment = " + totFAdj / numFAdj +
                "Num bias adjustments = " + numBAdj + ". Average bias adjustment = " + totBAdj / numBAdj);
    }

    // Return the dimensions of the output of this layer
    private void setOutDim(){
        outDim[0] = filters[0].length;
        outDim[1] = (inDim[1] - filterDim[0]) / vertStride + 1;
        outDim[2] = (inDim[2] - filterDim[1]) / horStride + 1;
    }

    // Given a scanner to a file, fill the parameters of this layer
    @Override
    public void fillParams(Scanner scIn){
        for (int i = 0; i < filters.length; i++){
            for (int j = 0; j < filters[0].length; j++){
                filters[i][j] = scIn.nextDouble();
            }
        }
        scIn.nextLine();
    }

    // Given a filewriter, write the layer by writing a metaline
    // containing -n for number of nodes, -a for activation function,
    // -d for pool dimensions, and -s for stride length. Precede these
    // options by writing a c for convolutionsl layer
    @Override
    public void writeLayer(FileWriter fWrite) throws IOException {
        StringBuilder outString = new StringBuilder();
        outString.append("conv -n ");
        outString.append(filters[0].length);
        outString.append(" -d ");
        outString.append(filterDim[0]);
        outString.append(" ");
        outString.append(filterDim[1]);
        outString.append(" -s ");
        outString.append(vertStride);
        outString.append(" ");
        outString.append(horStride);
        outString.append("\n");
        // output the parameters of the layer
        for (int i = 0; i < filters.length; i++){
            for (int j = 0; j < filters[0].length; j++){
                outString.append(filters[i][j]);
                outString.append(" ");
            }
            outString.append("\n");
        }
        fWrite.write(outString.toString());
    }
}
