// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.LossFunctions;

import Sequential.SequentialExceptions.InvalidDimensionException;

import java.io.FileWriter;
import java.io.IOException;

public class CrossEntropy implements LossFunction{
    @Override
    public double calculateLoss(double[] expected, double[] actual) throws InvalidDimensionException {
        if (expected.length != actual.length){
            throw new InvalidDimensionException("Expected values must be of the same dimensionality " +
                    "as the output of the network");
        }
        double total = 0;
        for (int i = 0; i < expected.length; i++){
            total += expected[i] * Math.log10(actual[i]);
        }
        return total * -1;
    }

    @Override
    public double[] calculatePDerivatives(double[] expected, double[] actual) throws InvalidDimensionException {
        if (expected.length != actual.length){
            throw new InvalidDimensionException("Expected values must be of the same dimensionality " +
                    "as the output of the network");
        }
        double result[] = new double[expected.length];
        for (int i = 0; i < result.length; i++){
            result[i] = -1 * expected[i] / (actual[i] * Math.log(10));
        }
        return result;
    }

    @Override
    public void writeFunc(FileWriter fWrite) throws IOException {
        fWrite.write("crossEntropy\n");
    }
}
