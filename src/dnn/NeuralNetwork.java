package dnn;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class NeuralNetwork {
  
  private Layer[] layers;
  private Layer outputLayer;
  private double sumSquaredErrors = 0;
  
  /* Initializations from Tom Mitchell */
  public double learningRate = 0.05;

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
  
  /**
   * Get a truth value and do back-propagation
   * 
   * @param outputTruthValue
   */
  public void backPropagateError(double[] outputTruthValue) {
    /**
     * According to Tom Mitchell, Machine Learning, Ch.4, p.98
     * 
     * 1.) For each network output unit k, calculate its error term sig[k]:
     * 
     * sig[k] = output[k] * (1 - output[k]) * (truth[k] - output[k])
     * 
     * where output[k] is the output of some neuron's activation function,
     * e.g. the sigmoid, and output[k] * (1 - output[k]) is the derivative
     * of the sigmoid function, and output[k] = sigmoid(k)
     * 
     * 2.) For each hidden unit h, calculate its error term sig[h]:
     * 
     * sig[h] = output[h] * (1 - output[h]) * SUM( weight[k][h] * sig[k] )
     * 
     * where k is the index of the sigmoid output[k] in the previous layer and each
     * weight[k][h] is one of the h incoming weights which are summed and sent through
     * the sigmoid.
     * 
     * 3.) Update each network weight w[j][i]
     * 
     * wNew[j][i] = w[j][i] - learningRate * sig[j] * x[j][i]
     * 
     * where x[j][i] is the input to the unit which is weighted by w[j][i]
     * 
     */
    
    
    /**
     * Output layer
     */
    for(int i = layers.length-1; i >= 0; i--) {
      Layer currLayer = layers[i];
      for(int j = 0; j < currLayer.getNumNeurons(); j++) {
        
      }
    }
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
