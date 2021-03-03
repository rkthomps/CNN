// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.SequentialExceptions;

/**
 * Exception is raised whenever an operation is performed out of order in the network.
 */
public class InvalidOperationException extends Exception {
    public InvalidOperationException(String message){
        super(message);
    }
}
