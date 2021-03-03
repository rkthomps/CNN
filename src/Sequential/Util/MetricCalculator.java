package Sequential.Util;

import Sequential.Layers.Layer;
import Sequential.LossFunctions.LossFunction;
import Sequential.SequentialExceptions.InvalidDimensionException;
import Sequential.SequentialExceptions.InvalidOperationException;

public class MetricCalculator {
    private Layer lastLayer;
    private LossFunction lossFunc;

    public MetricCalculator(Layer lastLayer, LossFunction lossFunc){
        this.lastLayer = lastLayer;
        this.lossFunc = lossFunc;
    }

    // Return the average loss given a batch of outputs
    public double calculateBatchLoss(double[][] expected)
            throws InvalidDimensionException, InvalidOperationException {
        double[][] lastAct = lastLayer.getLayerResult();
        if (lastAct == null){
            throw new InvalidOperationException("Batch loss cannot be calculated if there has not yet beed a foward pass");
        }
        if (lastAct.length != expected.length || lastAct[0].length != expected[0].length){
            throw new InvalidDimensionException("calculateBatchLoss: mismatch dimensions of expected vs actual");
        }
        double sum = 0;
        for (int i = 0; i < lastAct.length; i++){
            sum += lossFunc.calculateLoss(expected[i], lastAct[i]);
        }
        return sum / lastAct.length;
    }

    // Return the accuracy of a batch of data given the expected values
    public double calculateBatchAccuracy(double[][] expected) throws InvalidDimensionException{
        double[][] lastAct = lastLayer.getLayerResult();
        if (expected.length != lastAct.length || expected[0].length != lastAct[0].length){
            throw new InvalidDimensionException("cBatchAccuracy: Expected batch results and last layer activations" +
                    "must have the same dimensions.");
        }
        int numCorrect = 0;
        for (int i = 0; i < expected.length; i++){
            numCorrect += isCorrect(lastAct[i], expected[i]);
        }
        return numCorrect / (double)lastAct.length;
    }

    // Return the accuracy of an array of activations vs an array of expected values
    // the max value in the activations is the chosen value. Return 0 if not correct, 1 if correct
    public int isCorrect(double[] actual, double[] expected) throws InvalidDimensionException{
        if (actual.length != expected.length){
            throw new InvalidDimensionException("calculateAccuracy: length of actual and expected arrays must be identical");
        }
        // get the max of the actual array
        int argMax = -1;
        double max = Integer.MIN_VALUE;
        for (int i = 0; i < actual.length; i++){
            if (actual[i] > max){
                max = actual[i];
                argMax = i;
            }
        }
        return (int)expected[argMax];
    }
}
