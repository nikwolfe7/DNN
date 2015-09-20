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
    
  }

}
