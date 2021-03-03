package Sequential.Optimizers;

import Sequential.Sequential;
import Sequential.SequentialExceptions.InvalidDimensionException;
import Sequential.SequentialExceptions.InvalidOperationException;

import java.io.FileWriter;
import java.io.IOException;

public interface Optimizer {
    void trainNetwork(double[][][][] x, double[][] y, int batchSize, int epochs)
            throws InvalidOperationException, InvalidDimensionException;
    void constructOptimizer(Sequential net);
    void writeOpt(FileWriter fWrite) throws IOException;
}
