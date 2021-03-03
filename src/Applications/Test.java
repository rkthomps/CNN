package Applications;

import Sequential.SequentialExceptions.InvalidDimensionException;
import Sequential.Util.*;

import java.util.Arrays;
public class Test {


    public static void main(String[] args) throws InvalidDimensionException{
            System.out.println(Math.log(Math.E));
    }


    public static void main3(String[] args){
        int[][] inMap = new int[16][4];
        int[] filterDim = {2, 2};
        int[] inDim = {1, 4, 4};
        int vertStride = 1;
        int horStride = 1;
        int[] outDim = {1, 3, 3};
        // For each index in the flattened 3D input, store in columns the indexes that index is mapped to
        // if the index is mapped to less other indecies than the number of columns, the row will be padded
        // with -1
        // provide storage for each row of the inMap to indicate where to add the next location
        int[] curIndex = new int[inMap.length];
        int flatIndex;
        // fill matrix with -1, and the curIndex of each row with 0
        for (int i = 0; i < inMap.length; i++){
            curIndex[i] = 0;
            for (int j = 0; j < inMap[0].length; j++){
                inMap[i][j] = -1;
            }
        }

        // Iterate vertically (moving the filter)
        for (int i = 0; i + filterDim[0] <= inDim[1]; i += vertStride) {
            // Iterate horizontally (moving the filter)
            for (int j = 0; j + filterDim[1] <= inDim[2]; j += horStride) {
                // Iterate through layers of the filter
                for (int l = 0; l < inDim[0]; l++) {
                    // Iterate through rows of the filter
                    for (int r = 0; r < filterDim[0]; r++) {
                        // Iterate through columns of the filter
                        for (int c = 0; c < filterDim[1]; c++) {
                            flatIndex = (l * (inDim[1] * inDim[2]) + (r + i) * inDim[2] + (c + j));
                            inMap[flatIndex][curIndex[flatIndex]] = i * outDim[2] + j;
                            curIndex[flatIndex]++;
                        }
                    }
                }
            }
        }
        System.out.println(Arrays.deepToString(inMap));

    }
}

