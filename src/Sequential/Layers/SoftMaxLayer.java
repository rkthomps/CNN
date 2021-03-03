// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.Layers;

import Sequential.SequentialExceptions.InvalidDimensionException;
import Sequential.Util.NetUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class SoftMaxLayer extends Layer{

    public SoftMaxLayer(int[] inDim) throws InvalidDimensionException {
        super(inDim);
    }

    @Override
    public double[] forwardPass(double[] in) throws InvalidDimensionException {
        double sum = 0;
        double[] ret = new double[in.length];
        double curResult;

        // Compute exponents for each of the input scalars
        for (int i = 0; i < ret.length; i++){
            curResult = Math.pow(Math.E, in[i]);
            sum += curResult;
            ret[i] = curResult;
        }

        // Divide each value by the sum of the values
        for (int i = 0; i < ret.length; i++){
            ret[i] /= sum;
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


    // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    // the above website details how we can compute the jacobian for softmax functions
    // here I compute a 2d matrix representing the change in each of the softmax values with respect to each of the
    // inputs. Therefore it's size is inDim * inDDim
    @Override
    public double[][] computeGradients(double[][] jacob, double[][] prevInput) throws InvalidDimensionException {
        if (jacob.length != layerResult.length || jacob[0].length != layerResult[0].length){
            throw new InvalidDimensionException("The jacobian must be equal to the result of the layer in size");
        }
        double[][] softMaxGradientMatrix;
        double[][] ret = new double[jacob.length][jacob[0].length];
        for (int i = 0; i < jacob.length; i++){
            softMaxGradientMatrix = computeGradientMatrix(layerResult[i]);
            ret[i] = NetUtil.two2OneD(NetUtil.matMult(NetUtil.one2TwoD(jacob[i], 1, jacob[0].length), softMaxGradientMatrix));
        }
        return ret;
    }

    // Computes the gradients for each input to each softmax output
    private double[][] computeGradientMatrix(double[] softMaxInstance){
        double[][] ret = new double[softMaxInstance.length][softMaxInstance.length];
        for (int i = 0; i < ret.length; i++){
            for (int j = 0; j < ret[0].length; j++){
                if (i != j){
                    ret[i][j] =  -1 * softMaxInstance[i] * softMaxInstance[j];
                } else{
                    ret[i][j] = softMaxInstance[i] * (1 - softMaxInstance[j]);
                }
            }
        }
        return ret;
    }

    @Override
    public void writeLayer(FileWriter fWrite) throws IOException {
        String outStr = "softmax\n";
        fWrite.write(outStr);
    }

    @Override
    public void printInfo(){
        System.out.println("SoftMax Activation Layer");
    }
}
