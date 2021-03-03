package Sequential.Util;

import Sequential.Layers.Layer;
import Sequential.Layers.ReluLayer;
import Sequential.Layers.SigmoidLayer;
import Sequential.Layers.SoftMaxLayer;
import Sequential.SequentialExceptions.InvalidDimensionException;
import Sequential.SequentialExceptions.InvalidOperationException;

public class ActivationDisbatch {
    public static Layer getActFunc(String type, int[] inDim)
            throws InvalidOperationException, InvalidDimensionException {
        switch (type){
            case "sigmoid":
                return new SigmoidLayer(inDim);
            case "relu":
                return new ReluLayer(inDim);
            case "softmax":
                return new SoftMaxLayer(inDim);
            default:
                throw new InvalidOperationException("Activation function: " + type + " not found");
        }
    }
}