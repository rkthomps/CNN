package Sequential.Layers.TrainableLayer;

import Sequential.Layers.Layer;
import Sequential.SequentialExceptions.InvalidDimensionException;

import java.util.Scanner;

/**
 * A layer that has trainable parameters and that should have parameter update methods.
 */
public abstract class Trainable extends Layer {

    public Trainable(int[] inDim) throws InvalidDimensionException {
        super(inDim);
    }

    public abstract void printTrainInfo();
    public abstract void fillParams(Scanner scIn);
    public abstract void updateParamsMiniBatch(double learnRate) throws InvalidDimensionException;
    public abstract void updateParamsAdam(double alpha, double b1, double b2, double e);
}
