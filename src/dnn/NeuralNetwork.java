package dnn;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class NeuralNetwork {
  
  private Layer[] layers;
  private Layer outputLayer;

  public NeuralNetwork(Layer... layers) {
    this.layers = layers;
    this.outputLayer = layers[layers.length - 1];
  }
  
  public void feedForwardFromInput(double[] inputs) {
    for(Layer layer : layers) {
      layer.feedForward(inputs);
      inputs = layer.getLayerOutputActivations();
    }
  }

  /* Access internal layers */
  public Layer[] getLayers() {
    return layers;
  }
  
  /* get output from last layer */
  public double[] getNetworkOutput() {
    return outputLayer.getLayerOutputActivations();
  }
  
  /* Write learned weights to file */
  public void writeNetworkToFile(String fileName) throws IOException {
    FileWriter writer = new FileWriter(new File(fileName));
    for(Layer layer : layers)
      writer.write(layer.getStringRepresentation() + "\n");
    writer.close();
  }
  
  public int getNumInputs() {
    return layers[0].getNumInputs();
  }

  public int getNumOutputs() {
    return outputLayer.getNumNeurons();
  }
  
  public static void main(String[] args) throws IOException {
    double[] input = { 0.1, 0.4 };
    DNNFactory dnnFactory = new SimpleDNNFactory(input.length, 2, 4, 3, 2);
    NeuralNetwork network = dnnFactory.getInitializedNeuralNetwork();

    network.feedForwardFromInput(input);
    System.out.println("\nNetwork input:\t" + DNNUtils.printVector(input));
    System.out.println("Network output:\t" + DNNUtils.printVector(network.getNetworkOutput()));
    
    /* Write the network to file... */
    network.writeNetworkToFile("test.dnn.txt");
  }

  

}
