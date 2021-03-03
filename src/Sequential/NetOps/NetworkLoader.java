// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.NetOps;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import Sequential.Layers.TransformationLayer.TransformationLayer;
import Sequential.LossFunctions.LossFunction;
import Sequential.Optimizers.Adam;
import Sequential.Optimizers.MiniBatch;
import Sequential.Optimizers.Optimizer;
import Sequential.SequentialExceptions.*;
import Sequential.Sequential;
import Sequential.Util.*;
public class NetworkLoader {

    // Options for layer metadata
    Integer numNodes = null;
    int[] poolDim = null;
    int[] strideLength = null;

    // Save the network (weights and all) to a file with the given path
    // The file will have the following format:
    //      Inshape: x1 x2 x3
    //      Layer1: Metadata layer 1
    //      Layer1 weights and biases
    //      ...
    public Sequential loadNetwork(String path)
            throws FileNotFoundException, InvalidOperationException,
                InvalidDimensionException, InvalidNetworkFormatException {
        File fin = new File(path);
        Scanner scIn = new Scanner(fin);
        Sequential ret = new Sequential();
        String message;
        int curLayer = 0;


        // Get the input shape
        try{
            int[] inShape = getInShape(scIn);
            // Add Layers
            System.out.println("Loading layers.");
            while (addLayer(scIn, ret, inShape)) {
                curLayer++;
                System.out.println("Successfully loaded layer: " + curLayer);
                inShape = null;
            }
//            // net has not yet been compiled
//            if (ret.getMetrics() == null){
//                ret.compile();
//            }
            System.out.println("Loading complete");
        }
        catch (InvalidNetworkFormatException i){
            i.printStackTrace();
        }
        return ret;
    }

    // Read data from input scanner and try to add a layer to the network. Throw
    // an exception if the formatting is invald, return null if there is a
    // graceful end of file. Else, return the output dims of the successfully added layer
    private boolean addLayer(Scanner scIn, Sequential net, int[] inSize)
            throws InvalidNetworkFormatException, InvalidOperationException, InvalidDimensionException{
        if (!scIn.hasNextLine()){
            return false;
        }
        String[] line = scIn.nextLine().trim().split(" ");
        // Extra whitespace after last layer
        if (line[0].trim().length() == 0){
            return false;
        }
        switch (line[0]){
            case "conv":
                return addConvLayer(line, scIn, net, inSize);
            case "dense":
                return addDenseLayer(line, scIn, net, inSize);
            case "maxpool":
                return addMaxPoolLayer(line, scIn, net, inSize);
            case "relu":
                net.addRelu();
                return true;
            case "softmax":
                net.addSoftMax();
                return true;
            case "sigmoid":
                net.addSigmoid();
                return true;
            case "meanSquaredError":
            case "crossEntropy":
                compileNetwork(line, scIn, net);
                return false;
            default:
                throw new InvalidNetworkFormatException("Layer option: " + line[0] + " is not valid");
        }
    }

    // Add a convolutional layer to the network given the line of metadata and a scanner
    private boolean addConvLayer(String[] metaLine, Scanner scIn, Sequential net, int[] inSize)
            throws InvalidDimensionException, InvalidOperationException, InvalidNetworkFormatException {
        instanceToNull();
        setParams(metaLine);
        if (numNodes == null || poolDim == null){
            throw new InvalidNetworkFormatException("Convolutional layer must include num filters and filter size");
        }
        net.addConv(numNodes, poolDim, inSize, strideLength, null);
        ((TransformationLayer)net.getLastLayer()).fillParams(scIn);
        return true;
    }

    // Add a max pooling layer to the network given the line of metadata and a scanner
    private boolean addMaxPoolLayer(String[] metaLine, Scanner scIn, Sequential net, int[] inSize)
            throws InvalidDimensionException, InvalidOperationException, InvalidNetworkFormatException{
        instanceToNull();
        setParams(metaLine);
        if (poolDim == null){
            throw new InvalidNetworkFormatException("Maxpool layer must include num filters and filter size");
        }
        net.addMaxPool(poolDim, inSize, strideLength);
        return true;
    }

    // Add a convolutional layer to the network given the line of metadata and a scanner
    private boolean addDenseLayer(String[] metaLine, Scanner scIn, Sequential net, int[] inSize)
            throws InvalidDimensionException, InvalidOperationException, InvalidNetworkFormatException {
        instanceToNull();
        setParams(metaLine);

        if (numNodes == null){
            throw new InvalidNetworkFormatException("Dense layer must include num nodes");
        }
        net.addDense(numNodes, inSize, null);
        ((TransformationLayer)net.getLastLayer()).fillParams(scIn);
        return true;
    }

    // Given a layer of metadata, parse the options, and fill the given objects with the appropriate
    // data. The first token has the name of the layer. We need to start at the next token
    private void setParams(String[] metaLine)
            throws InvalidNetworkFormatException, InvalidOperationException{
        int curToken = 1;

        while (curToken < metaLine.length){
            switch(metaLine[curToken].trim()){
                case "-n":
                    if (curToken + 1 >= metaLine.length){
                        return;
                    }
                    numNodes = Integer.parseInt(metaLine[curToken + 1].trim());
                    curToken += 2;
                    break;
                case "-d":
                    if (curToken + 2 >= metaLine.length){
                        return;
                    }
                    poolDim = new int[2];
                    poolDim[0] = Integer.parseInt(metaLine[curToken + 1].trim());
                    poolDim[1] = Integer.parseInt(metaLine[curToken + 2].trim());
                    curToken += 3;
                    break;
                case "-s":
                    if (curToken + 2 >= metaLine.length){
                        return;
                    }
                    strideLength = new int[2];
                    strideLength[0] = Integer.parseInt(metaLine[curToken + 1].trim());
                    strideLength[1] = Integer.parseInt(metaLine[curToken + 2].trim());
                    curToken += 3;
                    break;
                default:
                    throw new InvalidNetworkFormatException("Invalid option: " + metaLine[curToken].trim());
            }
        }
    }

    // Use the given scanner to determine the input shape. It should be the next
    // line. If the next line is not a valid input shape, throw
    private int[] getInShape(Scanner scIn) throws InvalidNetworkFormatException{
        int[] ret = new int[3];
        if (!scIn.hasNextLine()){
            throw new InvalidNetworkFormatException("No input shape");
        }
        String[] tokens = scIn.nextLine().trim().split(" ");
        if (tokens.length != 4){
            throw new InvalidNetworkFormatException("Invalid # dim for input shape");
        }

        ret[0] = Integer.parseInt(tokens[1]);
        ret[1] = Integer.parseInt(tokens[2]);
        ret[2] = Integer.parseInt(tokens[3]);
        return ret;
    }

    // Adds either a loss function and metrics, jsut a loss function, or both's default to the net
    private void compileNetwork(String[] meta, Scanner scIn, Sequential net)
            throws InvalidNetworkFormatException, InvalidOperationException, InvalidDimensionException{
        Optimizer opt = constructOptimizer(scIn.nextLine().split(" "));
        LossFunction lossFunc = LossDisbatch.getLossFunc(meta[0]);
        String[] metrics = scIn.nextLine().split(" ");
        if (!metrics[0].equals("metrics:")){
            throw new InvalidNetworkFormatException("Invalid metrics line");
        }
        String[] compileMetrics = new String[metrics.length - 1];
        System.arraycopy(metrics, 1, compileMetrics, 0, metrics.length - 1);
        net.compile(lossFunc, opt, compileMetrics);
    }

    // given a line representing the serialization of an optimizer, return the optimizer
    // the line represents
    private Optimizer constructOptimizer(String[] tokens) throws InvalidNetworkFormatException{
        switch(tokens[0]){
            case "mini":
                return new MiniBatch(Double.parseDouble(tokens[1]));
            case "adam":
                return new Adam(Double.parseDouble(tokens[1]),
                        Double.parseDouble(tokens[2]),
                        Double.parseDouble(tokens[3]),
                        Double.parseDouble(tokens[4]));
            default:
                throw new InvalidNetworkFormatException("Invalid formatting for optimizer");
        }
    }

    // Set the value of the instance variables back to null
    private void instanceToNull(){
        numNodes = null;
        strideLength = null;
        poolDim = null;
    }

}
