// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.Optimizers;

import Sequential.Sequential;
import Sequential.SequentialExceptions.InvalidDimensionException;
import Sequential.SequentialExceptions.InvalidOperationException;

import java.io.FileWriter;
import java.io.IOException;

public interface Optimizer {
    /**
     * Fit the network to an approximate mapping between the given inputs and outputs.
     * @param x Array of 3D inputs to the network for training.
     * @param y Array of 1D expected outputs corresponding to the inputs.
     * @param batchSize Number of inputs that should be in training batches.
     * @param epochs Number of times the inputs should be trained on.
     * @throws InvalidDimensionException
     * @throws InvalidOperationException
     */
    void trainNetwork(double[][][][] x, double[][] y, int batchSize, int epochs)
            throws InvalidOperationException, InvalidDimensionException;

    /**
     * Gives the optimizer access to the network it needs to train.
     * @param net The network the optimizer should be fitting.
     */
    void constructOptimizer(Sequential net);

    /**
     * Serialize the optimizer.
     * @param fWrite The FileWriter that should write the serialized form of the optimizer.
     * @throws IOException
     */
    void writeOpt(FileWriter fWrite) throws IOException;
}
