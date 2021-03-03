package Sequential.LossFunctions;

import Sequential.SequentialExceptions.InvalidDimensionException;

import java.io.FileWriter;
import java.io.IOException;

public interface LossFunction {
    public double calculateLoss(double[] expected, double[] actual) throws InvalidDimensionException;
    public double[] calculatePDerivatives(double[] expected, double[] actual) throws InvalidDimensionException;
    public void writeFunc(FileWriter fWrite) throws IOException;
}
