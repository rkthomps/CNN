// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.LossFunctions;

import Sequential.SequentialExceptions.InvalidDimensionException;

import java.io.FileWriter;
import java.io.IOException;

public interface LossFunction {
    /**
     * Get the loss resulting from a single foward pass through the network.
     * @param expected Array of expected outputs.
     * @param actual Array of actual outputs.
     * @return The quantity of loss between the expected and actual outputs.
     * @throws InvalidDimensionException
     */
    public double calculateLoss(double[] expected, double[] actual) throws InvalidDimensionException;

    /**
     * Get the partial of the loss with respect to each of the outputs of the network.
     * @param expected Array of expected outputs.
     * @param actual Array of actual outputs.
     * @return Array of partial derivatives representing the change in loss with respect to that element.
     * @throws InvalidDimensionException
     */
    public double[] calculatePDerivatives(double[] expected, double[] actual) throws InvalidDimensionException;

    /**
     * Serialize the loss function to the file pointed to by the given FileWriter.
     * @param fWrite FileWriter object that should be written to.
     * @throws IOException
     */
    public void writeFunc(FileWriter fWrite) throws IOException;
}
