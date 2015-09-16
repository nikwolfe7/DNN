package dnn;

public class NeuralNetwork {
  
  private Layer[] layers;

  public NeuralNetwork(Layer... layers) {
    this.layers = layers;
  }
  
  public void feedForwardFromInput(double[] inputs) {
    for(Layer layer : layers) {
      layer.feedForward(inputs);
      inputs = layer.getLayerOutputActivations();
    }
  }
  
  public double[] getNetworkOutput() {
    /* get output from last layer */
    return layers[layers.length - 1].getLayerOutputActivations();
  }
  
  public static void main(String[] args) {
    
  }

}
