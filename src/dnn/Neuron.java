package dnn;

/**
 * Graph is here:
 * https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/ArtificialNeuronModel_english.png/600px-ArtificialNeuronModel_english.png
 * 
 * @author nwolfe
 */
public class Neuron {
  
  /* Neuron parameters */
  private volatile double[] weights;
  private volatile double weightedInputSum;
  private volatile double outputActivation;
  
  /* Initializations from Tom Mitchell */
  private double initLow = -0.05;
  private double initHigh = 0.05;

  /**
   * Number of inputs constructor randomly initializes all
   * weights between -1 and 1;
   *  
   * @param numInputs
   */
  public Neuron(int numInputs) {
    this.weightedInputSum = 0;
    this.outputActivation = 0;
    this.weights = new double[numInputs];
    for(int i = 0; i < weights.length; i++) {
      /* Generate random numbers between -1 and 1 */
      weights[i] = Math.random() * (initHigh - initLow) + initLow;
    }
  }
  
  /**
   * Constructor with initial weights
   * 
   * @param initialWeights
   */
  public Neuron(double[] initialWeights) {
    this.outputActivation = 0;
    this.weights = initialWeights;
  }
  
  public void replaceWeights(double[] replaceWeights) {
    this.weights = replaceWeights;
  }
  
  public void setActivation(double newActivation) {
    this.outputActivation = newActivation;
  }
  
  public void setWeightedInputSum(double weightedInputSum) {
    this.weightedInputSum = weightedInputSum;
  }

  public double[] getWeights() {
    return weights;
  }
  
  public double getActivation() {
    return outputActivation;
  }
  
  public double getWeightedInputSum() {
    return weightedInputSum;
  }

  public void excite(double[] input) {
    setWeightedInputSum(TransferFunction.weightedSum(input, getWeights()));
    setActivation(ActivationFunction.sigmoid(getWeightedInputSum()));
  }
  
  public String getStringRepresentation() {
    String[] arr = new String[weights.length];
    for(int i = 0; i < arr.length; i++)
      arr[i] = "" + weights[i];
    return "[" + String.join(",", arr) + "]";
  }
  
  public static Neuron instantiateFromString(String string) {
    String[] arr = string.split("\\,");
    double[] initialWeights = new double[arr.length];
    for(int i = 0; i < arr.length; i++)
      initialWeights[i] = Double.parseDouble(arr[i]);
    return new Neuron(initialWeights);
  }
  
  public static void main(String[] args) {
    Neuron n = new Neuron(5);  
    double[] impulse = new double[] {1,2,3,4,5};
    n.excite(impulse);
    System.out.println("Activation: " + n.getActivation());
  }

}
