package Sequential.WeightInitialization;

import Sequential.SequentialExceptions.InvalidOperationException;

import java.util.Random;

public class HeNormal {
    private double fanIn = 0;

    public void setIn(int fanIn){
        this.fanIn = fanIn;
    }
    public void setOut(int fanOut){}

    public double initializeWeight() throws InvalidOperationException {
        if (fanIn == 0){
            throw new InvalidOperationException("Try to initialize weights before setting # input nodes");
        }
        Random randomGen = new Random();
        return randomGen.nextGaussian() * Math.sqrt(2 / (fanIn));
    }
}
