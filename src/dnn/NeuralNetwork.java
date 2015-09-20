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
  
  public double[] getNetworkOutput() {
    /* get output from last layer */
    return outputLayer.getLayerOutputActivations();
  }
  
  public static void main(String[] args) {
    double[] input = { 0.1, 0.4, 0.5, 0.5, 0.23, 0.55, 0.7, 0.12, 0.111, 0.67 };
    DNNFactory dnnFactory = new SimpleDNNFactory(input.length, 2, 50, 40, 50);
    NeuralNetwork network = dnnFactory.getInitializedNeuralNetwork();

    network.feedForwardFromInput(input);
    
    StringBuilder sb = new StringBuilder("[  ");
    for(Double d : input)
      sb.append(d + "  ");
    System.out.println("\nNetwork input: " + sb.toString() + "]");
    
    sb = new StringBuilder("[  ");
    for(Double d : network.getNetworkOutput()) {
      sb.append(d + "  ");
    }
    System.out.println("Network output: " + sb.toString() + "]");
  }

}
