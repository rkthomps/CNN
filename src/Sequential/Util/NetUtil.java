package Sequential.Util;

import org.ejml.simple.SimpleMatrix;
import Sequential.SequentialExceptions.InvalidDimensionException;

public class NetUtil {

    // Use ejml lib to do efficient matrix multiplication
    public static double[][] matMult(double[][] m1, double[][] m2) throws InvalidDimensionException{
        if (m1[0].length != m2.length){
            throw new InvalidDimensionException("#Columns m1 must = #Rows m2");
        }
        SimpleMatrix a = new SimpleMatrix(m1);
        SimpleMatrix b = new SimpleMatrix(m2);
        SimpleMatrix result = a.mult(b);
        return one2TwoD(result.getDDRM().data, m1.length, m2[0].length);
    }

    // Perform elemnts wise matrix multiplication on the given two matricies, store changes in m1
    public static void elMulInc(double[][] m1, double[][] m2) throws InvalidDimensionException{
        if (m1.length != m2.length || m1[0].length != m2[0].length){
            throw new InvalidDimensionException("elMatMul: Dimensions of matricies to be multiplied must be identical");
        }
        for (int i = 0; i < m1.length; i++){
            for (int j = 0; j < m1[0].length; j++){
                m1[i][j]= m1[i][j] * m2[i][j];
            }
        }
    }

    // Perfore element wise matrix addition on the given two matricies, stroe changes in m1
    public static void elAddInc(double[][] m1, double[][] m2) throws InvalidDimensionException{
        if (m1.length != m2.length || m1[0].length != m2[0].length){
            throw new InvalidDimensionException("elMatMul: Dimensions of matricies to be multiplied must be identical");
        }
        for (int i = 0; i < m1.length; i++){
            for (int j = 0; j < m1[0].length; j++){
                m1[i][j]= m1[i][j] + m2[i][j];
            }
        }
    }

    // Return the transpose of the given matrix
    public static double[][] transpose(double[][] m){
        double[][] ret = new double[m[0].length][m.length];
        for (int i = 0; i < m.length; i++){
            for (int j = 0; j < m[0].length; j++){
                ret[j][i] = m[i][j];
            }
        }
        return ret;
    }

    // Take a sum of the columns of a matrix
    public static double[] colSum(double[][] m){
        double[] ret = new double[m[0].length];
        // Fill ret with 0
        for (int i = 0; i < ret.length; i++){
            ret[i] = 0;
        }

        // Accumulatie column values
        for (int i = 0; i < m.length; i++){
            for (int j = 0; j < m[0].length; j++){
                ret[j] += m[i][j];
            }
        }
        return ret;
    }

    // Return the element wise sum of two matricies
    public static double[][] matAdd(double[][] m1, double[][] m2) throws InvalidDimensionException{
        if (m1.length != m2.length || m1[0].length != m2[0].length){
            throw new InvalidDimensionException("Cannot add two arrays of different dimensions");
        }
        double[][] retArr = new double[m1.length][m1[0].length];
        for(int i = 0; i < m1.length; i++){
            for (int j = 0; j < m1[0].length; j++){
                retArr[i][j] = m1[i][j] + m2[i][j];
            }
        }
        return retArr;
    }

    // Given a oneD array, return a 2D array whose diagonal contains the elements
    // of the oneD array
    public static double[][] toDiag(double[] arr){
        double[][] ret = new double[arr.length][arr.length];
        for (int i = 0; i < ret.length; i++){
            ret[i][i] = arr[i];
        }
        return ret;
    }

    // Given a 2D array return a new 2d array with the same elements
    public static double[][] getCopy(double[][] in){
        double[][] ret = new double[in.length][in[0].length];
        for (int i = 0; i < ret.length; i++){
            System.arraycopy(in[i], 0, ret[i], 0, in[i].length);
        }
        return ret;
    }


    // Given a two dimensional array, return a one dimensional array
    public static double[] two2OneD(double[][] original){
        double[] retArr = new double[original.length * original[0].length];
        for (int i = 0; i < original.length; i++){
            System.arraycopy(original[i], 0, retArr, i * original[0].length, original[i].length);
        }
        return retArr;
    }

    // Given a one dimensional array, return a two dimensional array
    // with the given number of rows and columns
    public static double[][] one2TwoD(double[] original, int nRow, int nCol) throws InvalidDimensionException{
        if (original.length != nRow * nCol){
            throw new InvalidDimensionException("Number of elements in input and output matrix" +
                    "must be identical");
        }
        double[][] retArr = new double[nRow][nCol];
        for (int i = 0; i < nRow; i++){
            System.arraycopy(original, i * nCol, retArr[i], 0, retArr[i].length);
        }
        return retArr;
    }

    // Given a one dimensional arrya, return a three dimenstional array

    // Transforms a three dimensional double array into a 2 dimensional double array with the given
    // number of rows and columns by row major flattening
    public static double[][] three2TwoD(double[][][] in, int nRows, int nCols) throws InvalidDimensionException{
        if (in.length * in[0].length * in[0][0].length != nRows * nCols){
            throw new InvalidDimensionException("Number of input and output elements must be identical");
        }
        double[][] td = new double[nRows][nCols];
        int nElements = 0;
        for (int i = 0; i < in.length; i++){
            for (int j = 0; j < in[0].length; j++){
                for (int k = 0; k < in[0][0].length; k++){
                    td[nElements / nCols][nElements % nCols] = in[i][j][k];
                    nElements++;
                }
            }
        }
        return td;
    }

    // Transforms a three dimensional double array into a 1 dimensional double array by row major flattening
    public static double[] three2OneD(double[][][] in){
        double[] od = new double[in.length * in[0].length * in[0][0].length];
        int nElements = 0;
        for (int i = 0; i < in.length; i++){
            for (int j = 0; j < in[0].length; j++){
                for (int k = 0; k < in[0][0].length; k++){
                    od[nElements] = in[i][j][k];
                    nElements++;
                }
            }
        }
        return od;
    }

    // Transform a two dimensional array to a three dimensional array with the given dimensions
    public static double[][][] two2ThreeD(double[][] in, int z, int y, int x) throws InvalidDimensionException{
        if (in.length * in[0].length < z * y * x){
            throw new InvalidDimensionException("Number of input and output elements must be identical");
        }
        if (in.length * in[0].length > z * y * x){
            System.out.println("\nWarning: two2ThreeD: More elemnts in input than accounted for by output dim");
        }
        double[][][] ret = new double[z][y][x];
        int numElements = 0;
        for (int i = 0; i < z; i++){
            for (int j = 0; j < y; j++){
                for (int k = 0; k < x; k++){
                    ret[i][j][k] = in[numElements / in[0].length][numElements % in[0].length];
                    numElements++;
                }
            }
        }
        return ret;
    }

    // Get the min value of a 4d matrix
    public static double fourDMin(double[][][][] in){
        double min = Integer.MAX_VALUE;
        for (int i = 0; i < in.length; i++){
            for (int j = 0; j < in[0].length; j++){
                for (int k = 0; k < in[0][0].length; k++){
                    for (int l = 0; l < in[0][0][0].length; l++){
                        if (in[i][j][k][l] < min){
                            min = in[i][j][k][l];
                        }
                    }
                }
            }
        }
        return min;
    }
    // Get the min value of a 4d matrix
    public static double fourDMax(double[][][][] in){
        double max = Integer.MIN_VALUE;
        for (int i = 0; i < in.length; i++){
            for (int j = 0; j < in[0].length; j++){
                for (int k = 0; k < in[0][0].length; k++){
                    for (int l = 0; l < in[0][0][0].length; l++){
                        if (in[i][j][k][l] > max){
                            max = in[i][j][k][l];
                        }
                    }
                }
            }
        }
        return max;
    }

    // Return the location of the maximum of each row in a 2d array
    public static int[] argMaxByRow(double[][] in){
        int[] ret = new int[in.length];
        double max;
        int argMax;
        for (int i = 0; i < in.length; i++){
            argMax = -1;
            max = Integer.MIN_VALUE;
            for (int j = 0; j < in[0].length; j++){
               if (in[i][j] > max){
                   argMax = j;
                   max = in[i][j];
               }
            }
            ret[i] = argMax;
        }
        return ret;
    }

    // Perform min max normalization of the 4d input array
    public static void minMaxNormalize(double[][][][] in){
        double min = fourDMin(in);
        double max = fourDMax(in);
        for (int i = 0; i < in.length; i++){
            for (int j = 0; j < in[0].length; j++){
                for (int k = 0; k < in[0][0].length; k++){
                    for (int l = 0; l < in[0][0][0].length; l++){
                        in[i][j][k][l] = (in[i][j][k][l] - min) / (max - min);
                    }
                }
            }
        }

    }

    // Transfore a four dimensional arry into a five dimensional arry with the given dimensions
    public static double[][][][][] four2FiveD(double[][][][] in, int d1, int d2, int d3, int d4, int d5)
        throws InvalidDimensionException{
        if (in.length * in[0].length * in[0][0].length * in[0][0][0].length <
            d1 * d2 * d3 * d4 * d5){
            throw new InvalidDimensionException("Number of elements in input must at least be size of elemnts" +
                    "in ouput");
        }
        if (in.length * in[0].length * in[0][0].length * in[0][0][0].length >
                d1 * d2 * d3 * d4 * d5){
            System.out.println("\nWarning! four2FiveD: More elemnts in input than accounted for by output" +
                    " dimensions. Truncation will occur");
        }
        double[][][][][] ret = new double[d1][d2][d3][d4][d5];
        int d1Len = in[0].length * in[0][0].length * in[0][0][0].length;
        int d2Len = in[0][0].length * in[0][0][0].length;
        int d3Len = in[0][0][0].length;
        int numElements = 0;
        for (int i = 0; i < d1; i++){
            for (int j = 0; j < d2; j++){
                for (int k = 0; k < d3; k++){
                    for (int l = 0; l < d4; l++){
                        for (int m = 0; m < d5; m++){
                            ret[i][j][k][l][m] =
                                    in[numElements / d1Len][(numElements % d1Len) / d2Len]
                                    [(numElements % d1Len %d2Len) / d3Len]
                                            [numElements % d1Len % d2Len % d3Len];
                            numElements++;
                        }
                    }
                }
            }
        }
        return ret;
    }

    // Fill a 2D array with the given val
    public static void fill2D(double[][] in, double val){
        for (int i = 0; i < in.length; i++){
            for (int j = 0; j < in[0].length; j++){
                in[i][j] = val;
            }
        }
    }

    // Fill a 4D array with the given val
    public static void fill4D(double[][][][] in, double val){
        for (int i = 0; i < in.length; i++){
            for (int j = 0; j < in[0].length; j++){
                for (int k = 0; k < in[0][0].length; k++){
                    for (int l = 0; l < in[0][0][0].length; l++){
                        in[i][j][k][l] = val;
                    }
                }
            }
        }
    }
}
