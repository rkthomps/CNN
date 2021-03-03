// Author: Kyle Thompson
// Last Changed: 03/03/2021

package Sequential.SequentialExceptions;

/**
 * Exception is raised whenever there is a problem serializing the network, or reading the network serialization.
 */
public class InvalidNetworkFormatException extends Exception{
    public InvalidNetworkFormatException (String message){
        super(message);
    }
}
