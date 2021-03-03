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

public class MaxPool extends TransformationLayer {
    private int vertStride;
    private int horStride;
    private int[] outDim;
    private int[] poolDim;
    private int[][] preservedIn;
    private int[][][] inMap;

    // Initialize the size of the maxpool
    public MaxPool(int[] poolDim, int[] inDim, int[] strideLength) throws InvalidDimensionException{
        super(inDim);
        if (strideLength.length != 2){
            throw new InvalidDimensionException("Stride length array must be of length 2");
        }
        if (poolDim.length != 2){
            throw new InvalidDimensionException("Spacial window dimension array must be of length 2");
        }
        if (poolDim[0] > inDim[1] || poolDim[1] > inDim[2]){
            throw new InvalidDimensionException("Pool dimensions cannot be greater than layer input dimensions");
        }

        this.poolDim = Arrays.copyOf(poolDim, poolDim.length);
        vertStride = strideLength[0];
        horStride = strideLength[1];
        outDim = new int[3];
        setOutDim();
        inMap = new int[inDim[0] * inDim[1] * inDim[2]][(poolDim[0] / vertStride + 1) * (poolDim[1] / horStride + 1)][2];
        fillInMap();
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
        // Iterate through the layers of the input
        for (int layer = 0; layer < inDim[0]; layer++) {
            // Iterate vertically (moving the filter)
            for (int i = 0; i + poolDim[0] <= inDim[1]; i += vertStride) {
                // Iterate horizontally (moving the filter)
                for (int j = 0; j + poolDim[1] <= inDim[2]; j += horStride) {
                    // Iterate through rows of the filter
                    for (int r = 0; r < poolDim[0]; r++) {
                        // Iterate through columns of the filter
                        for (int c = 0; c < poolDim[1]; c++) {
                            // location of cur val in the input
                            flatIndex = (layer * (inDim[1] * inDim[2]) + (r + i) * inDim[2] + (c + j));
                            inMap[flatIndex][curIndex[flatIndex]][0] = (layer * outDim[1] * outDim[2] + (i / vertStride)
                                    * outDim[2] + (j / horStride));
                            inMap[flatIndex][curIndex[flatIndex]][1] = r * poolDim[1] + c;
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
            throw new InvalidDimensionException("MaxPool Layer: fowardPass: Input size doesn't match layer input dim");
        }
        double[][] ret = new double[outDim[0] * outDim[1] * outDim[2]][poolDim[0] * poolDim[1]];
        int j;

        // Assign each input value to its proper location in the formed input
        for (int i = 0; i < in.length; i++){
            j = 0;
            while (inMap[i][j][0] != -1){
                ret[inMap[i][j][0]][inMap[i][j][1]] = in[i];
                j++;
            }
        }
        return ret;
    }

     // Perform max pooling operations on a foward pass
    @Override
    public double[] forwardPass(double[] in) throws InvalidDimensionException{
        if (in.length != inDim[0] * inDim[1] * inDim[2]){
            throw new InvalidDimensionException("Given input dimensions don't match expected input dimensions");
        }
        double[][] formedIn = formedIn(in);
        double[] ret = new double[outDim[0] * outDim[1] * outDim[2]];
        int[] maxes = NetUtil.argMaxByRow(formedIn);
        for (int i = 0; i < ret.length; i++){
            ret[i] = formedIn[i][maxes[i]];
        }
        return ret;
    }

    // Perform max pooling operations on a foward pass
    @Override
    public double[][] forwardBatchPass(double[][] in) throws InvalidDimensionException{
        if (in[0].length != inDim[0] * inDim[1] * inDim[2]){
            throw new InvalidDimensionException("Given input dimensions don't match expected input dimensions");
        }
        preservedIn = new int[in.length][outDim[0] * outDim[1] * outDim[2]];
        layerResult = new double[in.length][outDim[0] * outDim[1] * outDim[2]];
        double[][] formedIn;
        int[] maxes;
        // for each batch
        for (int i = 0; i < in.length; i++){
            formedIn = formedIn(in[i]);
            maxes = NetUtil.argMaxByRow(formedIn);
            preservedIn[i] = maxes;
            // send maxes away
            for (int j = 0; j < layerResult[0].length; j++){
                layerResult[i][j] = formedIn[j][maxes[j]];
            }
        }
        return layerResult;
    }

    // Given the gradients of the loss with respect to this layer's output, compute and return the gradient of the
    // loss with respect to this layers input for all batches
    public double[][] computeGradients(double[][] jacob, double[][] prevInput) throws InvalidDimensionException{
        double[][] ret = new double[jacob.length][inDim[0] * inDim[1] * inDim[2]];
        double sum;
        int k;
        // For each batch
        for (int i = 0; i < jacob.length; i++){
            // For each input
            for (int j = 0; j < ret[0].length; j++){
                sum = 0;
                k = 0;
                while (inMap[j][k][0] != -1){
                    if (preservedIn[i][inMap[j][k][0]] == inMap[j][k][1]){
                        sum += jacob[i][inMap[j][k][0]];
                    }
                    k++;
                }
                ret[i][j] = sum;
            }
        }
        // TODO - SEE IF THIS ACTUALLY HELPS
        // preservedIn = null;
        return ret;
    }

    // Given the gradients of this layers weights, make appropriate adjustments. Nothing to do here. No params
    public void updateParamsMiniBatch(double learnRate) throws InvalidDimensionException{
    }

    // Update hte parameters for htis layer using an adam approach. Nothing to do here. no params
    public void updateParamsAdam(double alpha, double b1, double b2, double e){
    }

    // Return the output dimensions of this layer
    @Override
    public int[] getOutDim(){
        return outDim;
    }

    // Print the info for this layer
    @Override
    public void printInfo(){
        System.out.println(format("Maxpool Layer: Pool Size: " + Arrays.toString(poolDim) +
                        " In: " + Arrays.toString(inDim) + " Out: " + Arrays.toString(getOutDim()) + " HorStride: %4d VerStride: %4d",
                horStride, vertStride));
    }

    // Print the trining info for this layer
    @Override
    public void printTrainInfo(){
        System.out.println("Maxpool Layer. No parameters to be adjusted");
    }

    // Return the dimensions of the output of this layer
    private void setOutDim(){
        outDim[0] = inDim[0];
        outDim[1] = (inDim[1] - poolDim[0]) / vertStride + 1;
        outDim[2] = (inDim[2] - poolDim[1]) / horStride + 1;
    }

    // Doesn't do anything becuase a maxpool layer doesn't have parameters
    public void fillParams(Scanner scIn){}

    // Given a filewriter, write the layer by writing a metaline
    // containing -n for number of nodes, -a for activation function,
    // -d for pool dimensions, and -s for stride length. Precede these
    // options by writing a m for maxpool layer
    @Override
    public void writeLayer(FileWriter fWrite) throws IOException {
        String outString = "maxpool -d " + poolDim[0] + " " + poolDim[1] + " -s " + vertStride + " " + horStride + "\n";
        fWrite.write(outString);
    }
}
