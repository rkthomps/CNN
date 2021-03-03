import Sequential.LossFunctions.CrossEntropy;
import Sequential.Optimizers.Adam;
import Sequential.Optimizers.MiniBatch;
import Sequential.SequentialExceptions.*;
import Sequential.Util.*;
import UI.NeuralNetwork;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Scanner;

public class DigitRecognition {

    // Read in the data of a given scanner into an array of image labels
    // There will be a 2d array with the first dimension indicating the image, and
    // the next being the digit. the correct digit will be a 1 in an array of 10 with the rest 0
    private static double[][] readLabelFile(Scanner scIn){
        int numDim = scIn.nextInt();
        int dimSize = scIn.nextInt();
        int curDigit;
        double[][] labels = new double[dimSize][10];
        for (int i = 0; i < dimSize; i++){
            curDigit = scIn.nextInt();
            labels[i][curDigit] = 1;
        }
        return labels;
    }

    // Reads in the data of a given scanner into an array of flattened images
    // Result will be a 2D array with the rows of the images stacked against eachother
    private static double[][][][] readImagesFile(Scanner scIn){
        int numDim = scIn.nextInt();
        int numImages = scIn.nextInt();
        int numLayers = 1;
        int dim2Size = scIn.nextInt();
        int dim3Size = scIn.nextInt();

        double[][][][] images = new double[numImages][numLayers][dim2Size][dim3Size];
        for (int i = 0; i < images.length; i++) {
            for (int j = 0; j < images[0].length; j++){
                for (int k = 0; k < images[0][0].length; k++){
                    for (int l = 0; l < images[0][0][0].length; l++){
                        images[i][j][k][l] = scIn.nextInt();
                    }
                }
            }
        }
        return images;
    }

    public static void main(String[] Args)
            throws InvalidOperationException, InvalidDimensionException, IOException, InvalidNetworkFormatException {
        File fXTrain = new File("./data/trainingImages");
        File fYTrain = new File("./data/trainLabels");
        File fXTest = new File("./data/testingImages");
        File fYTest = new File("./data/testLabels");

        Scanner scXTrain = new Scanner(fXTrain);
        Scanner scYTrain = new Scanner(fYTrain);
        Scanner scXTest = new Scanner(fXTest);
        Scanner scYTest = new Scanner(fYTest);

        double[][][][] xTrain = readImagesFile(scXTrain);
        double[][] yTrain = readLabelFile(scYTrain);
        double[][][][] xTest = readImagesFile(scXTest);
        double[][] yTest = readLabelFile(scYTest);

        NetUtil.minMaxNormalize(xTrain);
        NetUtil.minMaxNormalize(xTest);

        int[] convDims= {3, 3};
        int[] poolDims = {2, 2};
        String[] metrics = {"accuracy"};
        int[] inputShape = {xTrain[0].length, xTrain[0][0].length, xTrain[0][0][0].length};

        System.out.println("Initializing Network");

        NeuralNetwork nn = new NeuralNetwork();
        nn.addConv(2, convDims, inputShape);
        nn.addMaxPool(poolDims);
        nn.addDense(64);
        nn.addDense(10);
        nn.compile(new CrossEntropy(), new Adam(), metrics);
        nn.fit(xTrain, yTrain, 64, 5);
        nn.evaluate(xTest, yTest);

    }
}

