package Sequential.Layers.TransformationLayer;

import Sequential.Layers.Layer;
import Sequential.SequentialExceptions.InvalidDimensionException;

import java.util.Scanner;

public abstract class TransformationLayer extends Layer {

    public TransformationLayer(int[] inDim) throws InvalidDimensionException {
        super(inDim);
    }

    public abstract void printTrainInfo();
    public abstract void fillParams(Scanner scIn);
    public abstract void updateParamsMiniBatch(double learnRate) throws InvalidDimensionException;
    public abstract void updateParamsAdam(double alpha, double b1, double b2, double e);
}
