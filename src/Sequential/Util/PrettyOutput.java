// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.Util;

import Sequential.Layers.Layer;
import Sequential.LossFunctions.LossFunction;
import Sequential.SequentialExceptions.InvalidDimensionException;
import Sequential.SequentialExceptions.InvalidOperationException;
import Sequential.Util.MetricCalculator;

import static java.lang.String.format;

public class PrettyOutput {
    private int epochs;
    private int batchesCompleted;
    private int numBatches;
    private double curProgress;
    private int numStars;
    private boolean calcAccuracy;
    private double totLoss;
    private double totAccuracy;
    private MetricCalculator mc;

    public PrettyOutput(int numBatches, int epochs, int numStars, Layer lastLayer,
                        String[] metrics, LossFunction lossFunc){
        this.epochs = epochs;
        this.numBatches = numBatches;
        this.numStars = numStars;

        for (int i = 0; i < metrics.length; i++){
            if (metrics[i].equalsIgnoreCase("accuracy")){
                calcAccuracy = true;
                break;
            }
        }
        mc = new MetricCalculator(lastLayer, lossFunc);
    }

    // Print the start of an epoch line
    public void newEpoch(int curEpoch){
        batchesCompleted = 0;
        curProgress = 0;
        totLoss = 0;
        totAccuracy = 0;
        System.out.print(format("Epoch: (%d/%d) ", curEpoch + 1, epochs));
    }

    // Prints stars to indicate completed batches
    public void logProgress(double[][] expected) throws InvalidDimensionException, InvalidOperationException {
        batchesCompleted++;
        if (batchesCompleted / ((double) numBatches) * 100 > curProgress + (100.0 / numStars)){
            System.out.print("*");
            curProgress = batchesCompleted / ((double) numBatches) * 100;
        }
        totLoss += mc.calculateBatchLoss(expected);
        if (calcAccuracy){
            totAccuracy += mc.calculateBatchAccuracy(expected);
        }
    }

    // Prints the loss for the given epoch
    public void finishEpoch(){
        System.out.print(format(" Loss = %5f", totLoss / numBatches));
        if (calcAccuracy){
            System.out.print(format(" Accuracy = %5f", totAccuracy / numBatches));
        }
        System.out.println();
    }
}
