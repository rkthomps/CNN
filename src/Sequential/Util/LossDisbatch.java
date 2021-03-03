package Sequential.Util;

import Sequential.LossFunctions.CrossEntropy;
import Sequential.LossFunctions.LossFunction;
import Sequential.LossFunctions.MeanSquaredError;
import Sequential.SequentialExceptions.InvalidNetworkFormatException;

public class LossDisbatch {
    public static LossFunction getLossFunc(String lossFunc) throws InvalidNetworkFormatException{
        switch (lossFunc){
            case "meanSquaredError":
                return new MeanSquaredError();
            case "crossEntropy":
                return new CrossEntropy();
            default:
                throw new InvalidNetworkFormatException("Loss function " + lossFunc + " not found");
        }
    }
}
