// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.Layers;

import Sequential.SequentialExceptions.InvalidDimensionException;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

public abstract class Layer {
    protected int[] inDim;
    protected double[][] layerResult;

    public Layer(int[] inDim) throws InvalidDimensionException{
        if (inDim.length != 3){
            throw new InvalidDimensionException("Incoming dimension sizes must be of length 3");
        }
        this.inDim = Arrays.copyOf(inDim, inDim.length);
    }

    // Return the result of passing a batch of data through the layer
    public  double[][] getLayerResult(){
        return layerResult;
    }
    // Return the outgoing dimensions of the layer. For activation layers this is trivial
    // this will be overridden in the transformation layer extending classes
    public int[] getOutDim(){
        return inDim;
    };

    public abstract void printInfo();
    public abstract double[] forwardPass(double[] in) throws InvalidDimensionException;
    public abstract double[][] forwardBatchPass(double[][] in) throws InvalidDimensionException;
    public abstract double[][] computeGradients(double[][] jacob, double[][] prevInput) throws InvalidDimensionException;
    public abstract void writeLayer(FileWriter fWrite) throws IOException;

}
