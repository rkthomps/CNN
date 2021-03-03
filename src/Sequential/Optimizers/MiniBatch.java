// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.Optimizers;

import Sequential.Layers.TransformationLayer.TransformationLayer;
import Sequential.Sequential;
import Sequential.Util.*;
import Sequential.Layers.Layer;
import Sequential.SequentialExceptions.InvalidDimensionException;
import Sequential.SequentialExceptions.InvalidOperationException;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class MiniBatch implements Optimizer{
    private double learnRate;
    private Sequential net;

    public MiniBatch() {this.learnRate = 0.01;}
    public MiniBatch(double learnRate){
        this.learnRate = learnRate;
    }

    // Provide more construction parameters for this optimizer
    public void constructOptimizer(Sequential net){
        this.net = net;
    }

    // Train this network
    public void trainNetwork(double[][][][] in, double[][] out, int batchSize, int epochs)
            throws InvalidOperationException, InvalidDimensionException {

        double[][][][][] x =
                NetUtil.four2FiveD(in, in.length / batchSize, batchSize, in[0].length, in[0][0].length, in[0][0][0].length);
        double[][][] y =
                NetUtil.two2ThreeD(out, out.length / batchSize, batchSize, out[0].length);

        ArrayList<Layer> layers = net.getLayers();
        PrettyOutput pOutput = new PrettyOutput(x.length, epochs, 30, net.getLastLayer(), net.getMetrics(),
                net.getLossFunc());
        double[][] curGradients;
        double[][][] flatIn = toFlatBatches(x);

        // Train over the given number of epochs
        for (int curEpoch = 0; curEpoch < epochs; curEpoch++) {
            pOutput.newEpoch(curEpoch);
            // Train over each batch
            for (int i = 0; i < x.length; i++){
                net.forwardBatchPass(flatIn[i]);
                curGradients = net.getLossJacobian(y[i]);
                pOutput.logProgress(y[i]);
                for (int j = layers.size() - 1; j >= 0; j--){
                    if (j > 0){
                        curGradients = layers.get(j).computeGradients(curGradients, layers.get(j - 1).getLayerResult());
                    }
                    else{
                        curGradients = layers.get(j).computeGradients(curGradients, flatIn[i]);
                    }
                    if (layers.get(j) instanceof TransformationLayer){
                        ((TransformationLayer) layers.get(j)).updateParamsMiniBatch(learnRate);
                    }
                }
            }
            pOutput.finishEpoch();
        }
    }

    // Takes the five dimensional array used for training and create a three dimensional array with the first dimension
    // being each batch, the next being the inputs in the batch, and the next being the actual values
    private double[][][] toFlatBatches(double[][][][][] in){
        double[][][] flatIn = new double[in.length][in[0].length][in[0][0].length * in[0][0][0].length * in[0][0][0][0].length];
        // Flatten each input
        for (int i = 0; i < flatIn.length; i++){
            for (int j = 0; j < flatIn[0].length; j++){
                flatIn[i][j] = NetUtil.three2OneD(in[i][j]);
            }
        }
        return flatIn;
    }

    // Output information about this optimizer in the serialization format
    public void writeOpt(FileWriter fWrite) throws IOException{
        StringBuilder sb = new StringBuilder("mini ");
        sb.append(learnRate);
        sb.append("\n");
        fWrite.write(sb.toString());
    }
}
