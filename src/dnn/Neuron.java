package dnn;

/**
 * Graph is here:
 * https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/ArtificialNeuronModel_english.png/600px-ArtificialNeuronModel_english.png
 * 
 * @author nwolfe
 */
public class Neuron {
  
  private volatile double[] weights;
  private volatile double activation;

  /**
   * Number of inputs constructor randomly initializes all
   * weights between 0 and 1;
   *  
   * @param numInputs
   */
  public Neuron(int numInputs) {
    this.activation = 0;
    this.weights = new double[numInputs];
    for(int i = 0; i < weights.length; i++) {
      weights[i] = Math.random();
    }
  }
  
  /**
   * Constructor with initial weights
   * 
   * @param initialWeights
   */
  public Neuron(double[] initialWeights) {
    this.activation = 0;
    this.weights = initialWeights;
  }
  
  public void replaceWeights(double[] replaceWeights) {
    this.weights = replaceWeights;
  }
  
  public void setActivation(double newActivation) {
    this.activation = newActivation;
  }
  
  public double[] getWeights() {
    return weights;
  }
  
  public double getActivation() {
    return activation;
  }

  public void excite(double[] input) {
    double weightedSum = TransferFunction.weightedSum(input, getWeights());
    setActivation(ActivationFunction.sigmoid(weightedSum));
  }
  
  public static void main(String[] args) {
    Neuron n = new Neuron(5);  
    double[] impulse = new double[] {1,2,3,4,5};
    n.excite(impulse);
    System.out.println("Activation: " + n.getActivation());
  }

}