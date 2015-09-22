package dnn;

import java.util.Arrays;
import java.util.regex.Pattern;

public class Layer {
  
  private volatile double[] layerActivations;
  private Neuron[] neurons;
  private int numInputs;
  private double biasInput = 1;
  private double[] inputArray;

  /**
   * Allow the layer to set up the neurons for you...
   * 
   * @param numNeurons
   * @param numInputs
   */
  public Layer(int numNeurons, int inputDimension) {
    /* adding the bias input */
    this.numInputs = inputDimension + 1;
    /* create an array with all 1's, then copy input to it */
    this.inputArray = new double[numInputs];
    Arrays.fill(inputArray, biasInput);
    this.neurons = new Neuron[numNeurons];
    /* neurons have random initialization */
    for (int i = 0; i < neurons.length; i++)
      neurons[i] = new Neuron(numInputs);
  }
  
  /**
   * If the neurons have been preconfigured, assuming
   * they are pre-trained with optimal weights, etc...
   * 
   * @param neurons
   */
  public Layer(Neuron... neurons) {
    this.neurons = neurons;
    this.numInputs = neurons[0].getWeights().length;
    this.inputArray = new double[numInputs];
    Arrays.fill(inputArray, biasInput);
  }
  
  public int getNumNeurons() {
    return neurons.length;
  }
  
  /* Number of inputs to layer without bias */
  public int getNumInputs() {
    return numInputs - 1;
  }
  
  private void setLayerActivations(double[] activations) {
    this.layerActivations = activations;
  }
  
  public double[] getLayerOutputActivations() {
    return layerActivations;
  }
  
  public void feedForward(double[] input) {
    /* copy to inputArray containing bias input */
    System.arraycopy(input, 0, inputArray, 0, input.length);
    double[] activations = new double[neurons.length];
    for(int i = 0; i < neurons.length; i++) {
      neurons[i].excite(inputArray);
      activations[i] = neurons[i].getActivation();
    }
    setLayerActivations(activations);
  }
  
  /**
   * Retrieve layer weights as a matrix
   * 
   * @return
   */
  public double[][] getLayerWeights() {
    double[][] layerWeights = new double[neurons.length][numInputs];
    for (int i = 0; i < layerWeights.length; i++)
      layerWeights[i] = neurons[i].getWeights();
    return layerWeights;
  }
  
  /**
   * Set layer weights as a matrix
   */
  public void updateLayerWeights(double[][] newWeights) {
    for (int i = 0; i < newWeights.length; i++)
      neurons[i].replaceWeights(newWeights[i]);
  }
  
  /**
   * get string representation to write to file
   * @return String
   */
  public String getStringRepresentation() {
    String[] arr = new String[neurons.length];
    for(int i = 0; i < arr.length; i++)
      arr[i] = neurons[i].getStringRepresentation();
    return "[" + String.join(",", arr) + "]";
  }
  
  public static Layer instantiateFromString(String string) {
    string = string.replaceAll(Pattern.quote("[["),"").replaceAll(Pattern.quote("]]"),"");
    String[] arr = string.split(Pattern.quote("],["));
    Neuron[] neurons = new Neuron[arr.length];
    for(int i = 0; i < arr.length; i++)
      neurons[i] = Neuron.instantiateFromString(arr[i]);
    return new Layer(neurons);
  }
  
  public static void main(String[] args) {
    Layer l1 = new Layer(4, 3);
    double[] input = { 0.1, 0.2, 0.3 };
    l1.feedForward(input);
    StringBuilder sb = new StringBuilder("[");
    for(Double d : l1.getLayerOutputActivations()) {
      sb.append(d + ",");
    }
    sb.append("]");
    System.out.println("Layer output: " + sb.toString());
  }

}
