// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.Layers;

import Sequential.SequentialExceptions.InvalidDimensionException;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class SigmoidLayer extends Layer{

    public SigmoidLayer(int[] inDim) throws InvalidDimensionException{
        super(inDim);
    }

    @Override
    public double[] forwardPass(double[] in) throws InvalidDimensionException {
        double[] ret = new double[in.length];
        for (int i = 0; i < ret.length; i++){
            ret[i] = (1 / (1 + Math.pow(Math.E, -1 * in[i])));
        }
        return ret;
    }

    @Override
    public double[][] forwardBatchPass(double[][] in) throws InvalidDimensionException {
        layerResult = new double[in.length][in[0].length];
        for (int i = 0; i < in.length; i++){
            layerResult[i] = forwardPass(in[i]);
        }
        return layerResult;
    }

    @Override
    public double[][] computeGradients(double[][] jacob, double[][] prevInput) throws InvalidDimensionException {
        if (prevInput.length != jacob.length || prevInput[0].length != jacob[0].length){
            throw new InvalidDimensionException("In an activation layer, size of jacobian must equal size of previous input");
        }
        double[][] ret = new double[prevInput.length][prevInput[0].length];
        for (int i = 0; i < ret.length; i++){
            for (int j = 0; j < ret[0].length; j++){
                ret[i][j] = Math.pow(Math.E, -1 * prevInput[i][j]) / (Math.pow(1 + Math.pow(Math.E, -1 * prevInput[i][j]), 2));
            }
        }
        return ret;
    }

    @Override
    public void writeLayer(FileWriter fWrite) throws IOException {
        String outString = "sigmoid\n";
        fWrite.write(outString);
    }

    @Override
    public void printInfo(){
        System.out.println("Sigmoid Activation Layer");
    }
}
