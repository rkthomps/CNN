package Sequential.NetOps;

import Sequential.Layers.Layer;
import Sequential.Sequential;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class NetworkWriter {

    private Sequential net;

    public NetworkWriter(Sequential net){
        this.net = net;
    }

    public void writeNetwork(String path)
            throws IOException {
        File outFile = new File(path);
        FileWriter fWrite = new FileWriter(outFile);
        writeInShape(fWrite);
        ArrayList<Layer> layers = net.getLayers();
        String[] metrics = net.getMetrics();
        StringBuilder metricsOut = new StringBuilder();
        // Write Layers
        for (int i = 0; i < layers.size(); i++){
            System.out.println("Writing layer: " + (i + 1) + "/" + layers.size());
            layers.get(i).writeLayer(fWrite);
        }
        // Write Loss
        net.getLossFunc().writeFunc(fWrite);
        // Write Optimizer
        net.getOptimizer().writeOpt(fWrite);
        // Write metrics
        metricsOut.append("metrics: ");
        for (int i = 0; i < metrics.length; i++){
            metricsOut.append(metrics[i]);
            metricsOut.append(" ");
        }
        metricsOut.append("\n");
        fWrite.write(metricsOut.toString());
        fWrite.close();
        System.out.println("Writing complete.");
    }

    // Write the input dimensions to the network to a file
    private void writeInShape(FileWriter fWrite) throws IOException{
        int[] inDim = net.getInDim();
        String outStr = "Inshape: " + inDim[0] + " " + inDim[1] + " " + inDim[2] + "\n";
        fWrite.write(outStr);
    }
}
