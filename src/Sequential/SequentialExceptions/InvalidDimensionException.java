// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.SequentialExceptions;

/**
 * Exception is raised whenever there is a conflict in dimensionality during a network operation.
 */
public class InvalidDimensionException extends Exception{
    public InvalidDimensionException(String message) {super(message);}
}
