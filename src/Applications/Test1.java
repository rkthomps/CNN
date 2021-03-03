package Applications;

import Sequential.LossFunctions.MeanSquaredError;
import Sequential.SequentialExceptions.*;
import Sequential.Sequential;

import java.util.Arrays;

public class Test1 {
    public static void main(String[] args){
        int[] arr1 = {1, 2, 3};
        int[] arr2 = new int[4];

        System.arraycopy(arr1, 0, arr2, 0, 3);
        arr2[arr2.length - 1] = 10;
        System.out.println(Arrays.toString(arr2));


    }



}
