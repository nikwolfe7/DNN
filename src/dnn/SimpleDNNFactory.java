package dnn;

public class SimpleDNNFactory implements DNNFactory {

  private NeuralNetwork network;

  /**
   * Takes a variable number of parameters... 
   * (NOTE: output layer is separate from hidden layers!)
   * 
   * @param inputs - dimensionality of the inputs
   * @param outputs - dimensionality of the outputs / output layer
   * @param hiddenLayerDimensions - dimensionality of the hidden layers -- IN ORDER!
   */
  public SimpleDNNFactory(Integer inputs, Integer outputs, Integer... hiddenLayerDimensions) {
    StringBuilder sb = new StringBuilder();
    for(int i = 0; i < hiddenLayerDimensions.length; i++)
      sb.append("| Hidden Layer " + (i+1) + ":\t" + hiddenLayerDimensions[i] + " Neurons\n");
    System.out.print("DNN Factory"
            + "\n------------------------------------------------------"
            + "\nBuilding neural network with the following dimensions:"
            + "\n| Number of Inputs:\t" + inputs
            + "\n| Number of Outputs:\t" + outputs
            + "\n" + sb.toString()
            + "\nInitializing... ");
    /* Number of layers is the number of hidden layers plus the output layer */
    Layer[] layers = new Layer[hiddenLayerDimensions.length + 1];
    /* Layer takes 2 params: # neurons and # inputs from previous layer */
    layers[0] = new Layer(hiddenLayerDimensions[0], inputs); // input layer
    /* Hidden layers */
    for (int i = 1; i < hiddenLayerDimensions.length; i++) {
      int prevLayerInputs = layers[i - 1].getNumNeurons();
      layers[i] = new Layer(hiddenLayerDimensions[i], prevLayerInputs);
    }
    /* Output layers */
    int outputLayer = layers.length - 1;
    int prevLayerInputs = layers[outputLayer - 1].getNumNeurons();
    layers[outputLayer] = new Layer(outputs, prevLayerInputs);
    this.network = new NeuralNetwork(layers);
    System.out.println("Done!");
  }

  @Override
  public NeuralNetwork getInitializedNeuralNetwork() {
    return network;
  }

  public static void main(String[] args) {
    /* Simple network with 3 inputs, 1 output, and a single hidden layer of width 2 */
    DNNFactory dnnFactory = new SimpleDNNFactory(3, 1, 5, 4, 5);
    NeuralNetwork network = dnnFactory.getInitializedNeuralNetwork();
  }

}
