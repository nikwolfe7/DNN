package dnn;

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
  
  public int getNumInputs() {
    return layers[0].getNumInputs();
  }

  public int getNumOutputs() {
    return outputLayer.getNumNeurons();
  }
  
  public static void main(String[] args) {
    double[] input = { 0.1, 0.4, 0.5, 0.5, 0.23, 0.55, 0.7, 0.12, 0.111, 0.67 };
    DNNFactory dnnFactory = new SimpleDNNFactory(input.length, 2, 50, 40, 50);
    NeuralNetwork network = dnnFactory.getInitializedNeuralNetwork();

    network.feedForwardFromInput(input);
    System.out.println("\nNetwork input:\t" + DNNUtils.printVector(input));
    System.out.println("Network output:\t" + DNNUtils.printVector(network.getNetworkOutput()));
  }

  

}
