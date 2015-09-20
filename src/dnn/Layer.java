package dnn;

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
    this.inputArray = new double[numInputs];
    /* create an array with all 1's, then copy input to it */
    for (int i = 0; i < inputArray.length; i++)
      inputArray[i] = biasInput;
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
    this.numInputs = neurons[0].getWeights().length;
    this.neurons = neurons;
  }
  
  public int getNumNeurons() {
    return neurons.length;
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
