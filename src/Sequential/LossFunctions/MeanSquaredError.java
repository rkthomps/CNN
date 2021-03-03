// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.LossFunctions;

import Sequential.SequentialExceptions.InvalidDimensionException;

import java.io.FileWriter;
import java.io.IOException;

public class MeanSquaredError implements LossFunction {

    public double calculateLoss (double[] expected, double[] actual) throws InvalidDimensionException{
        if (expected.length != actual.length){
            throw new InvalidDimensionException("Expected values must be of the same dimensionality " +
                    "as the output of the network");
        }
        double totLoss = 0;
        for (int i = 0; i < expected.length; i++){
            totLoss += Math.pow((expected[i] - actual[i]), 2);
        }
        return totLoss / expected.length;
    }

    public double[] calculatePDerivatives (double[] expected, double[] actual) throws InvalidDimensionException{
        if (expected.length != actual.length){
            throw new InvalidDimensionException("Expected values must be of the same dimensionality " +
                    "as the output of the network");
        }
        double[] dMSE = new double[expected.length];
        for (int i = 0; i < dMSE.length; i++){
           dMSE[i] = -2 * (expected[i] - actual[i]) / expected.length;
        }
        return dMSE;
    }

    public void writeFunc(FileWriter fWrite) throws IOException {
        fWrite.write("meanSquaredError\n");
    }
}
