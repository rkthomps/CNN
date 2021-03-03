// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.Optimizers;

import Sequential.Layers.Layer;
import Sequential.Layers.TransformationLayer.TransformationLayer;
import Sequential.Sequential;
import Sequential.SequentialExceptions.InvalidDimensionException;
import Sequential.SequentialExceptions.InvalidOperationException;
import Sequential.Util.NetUtil;
import Sequential.Util.PrettyOutput;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Class representing an optimizer that implements the Adam optimization algorithm.
 */
public class Adam implements Optimizer{
    private double alpha;
    private double beta1;
    private double beta2;
    private double epsilon;
    private Sequential net;

    /**
     * Constructs a new Adam Optimizer.
     * @param alpha Initial learning rate to be used by the optimizer.
     * @param beta1 Initial first moment momentum to be used by the optimizer.
     * @param beta2 Initial second moment momentum to be used by the optimizer.
     * @param epsilon Initial epsilon value to be used by the optimizer.
     */
    public Adam(double alpha, double beta1, double beta2, double epsilon){
        this.alpha = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    /**
     * Constructs a new Adam Optimizer.
     * @param alpha Initial learning rate to be used by the optimizer.
     * @param beta1 Initial first moment momentum to be used by the optimizer.
     * @param beta2 Initial second moment momentum to be used by the optimizer.
     */
    public Adam(double alpha, double beta1, double beta2){
        this.alpha = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = Math.pow(10, -8);
    }

    /**
     * Constructs a new Adam Optimizer.
     * @param beta1 Initial first moment momentum to be used by the optimizer.
     * @param beta2 Initial second moment momentum to be used by the optimizer.
     */
    public Adam(double beta1, double beta2){
        this.alpha = 0.001;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = Math.pow(10, -8);
    }

    /**
     * Constructs a new Adam Optimizer.
     * @param alpha Initial learning rate to be used by the optimizer.
     */
    public Adam(double alpha){
        this.alpha = alpha;
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.epsilon = Math.pow(10, -8);
    }

    /**
     * Constructs a new Adam Optimizer.
     */
    public Adam(){
        this.alpha = 0.001;
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.epsilon = Math.pow(10, -8);
    }

    // Provide more construction parameters for this optimizer
    public void constructOptimizer(Sequential net){
        this.net = net;
    }

    // Train this network
    public void trainNetwork(double[][][][] in, double[][] out, int batchSize, int epochs)
            throws InvalidOperationException, InvalidDimensionException {
        double curBeta1;
        double curBeta2;
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
            curBeta1 = Math.pow(beta1, curEpoch + 1);
            curBeta2 = Math.pow(beta2, curEpoch + 1);
            // Train over each batch
            for (int i = 0; i < x.length; i++) {
                net.forwardBatchPass(flatIn[i]);
                curGradients = net.getLossJacobian(y[i]);
                pOutput.logProgress(y[i]);
                for (int j = layers.size() - 1; j >= 0; j--) {
                    if (j > 0) {
                        curGradients = layers.get(j).computeGradients(curGradients, layers.get(j - 1).getLayerResult());
                    } else {
                        curGradients = layers.get(j).computeGradients(curGradients, flatIn[i]);
                    }
                    if (layers.get(j) instanceof TransformationLayer){
                        ((TransformationLayer) layers.get(j)).updateParamsAdam(alpha, curBeta1, curBeta2, epsilon);
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

    // Write information about this optimizer in the serialization format
    public void writeOpt(FileWriter fWrite) throws IOException{
        StringBuilder sb = new StringBuilder("adam ");
        sb.append(alpha);
        sb.append(" ");
        sb.append(beta1);
        sb.append(" ");
        sb.append(beta2);
        sb.append(" ");
        sb.append(epsilon);
        sb.append("\n");
        fWrite.write(sb.toString());
    }
}
