package main;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

import dnn.DNNFactory;
import dnn.DNNUtils;
import dnn.NeuralNetwork;
import dnn.SimpleDNNFactory;

public class Driver {
  
  public static void main(String[] args) throws FileNotFoundException, IOException {
    int numInputs = 26;
    int numDataInstances = 5;
    int numOutputs = 1;
    List<double[]> data = DNNUtils.getInputsFromFile(DataGenerator.getAdditionData(numInputs, numDataInstances));
    DNNFactory factory = new SimpleDNNFactory(numInputs, numOutputs, 250, 25, 250, 25, 300, 60, 250, 25);
    trainDNN(data, factory.getInitializedNeuralNetwork());
  }
  
  private static void trainDNN(List<double[]> data, NeuralNetwork network) {
    int inputs, outputs;
    inputs = network.getNumInputs();
    outputs = network.getNumOutputs();
    double[] inputData = new double[inputs];
    double[] outputData = new double[outputs];
    double squaredError = 0;
    for(double[] instance : data) {
      /* Copy section of instance belonging to the label/truth value */
      System.arraycopy(instance, 0, outputData, 0, outputs);
      /* Copy section of instance belonging to the input vector */
      System.arraycopy(instance, outputs, inputData, 0, inputs);
      /* Run the data through the network */
      network.feedForwardFromInput(inputData);
      /* Output */
      System.out.println("Input:\t" + DNNUtils.printVector(inputData));
      System.out.println("Output:\t" + DNNUtils.printVector(network.getNetworkOutput()));
      System.out.println("Truth:\t" + DNNUtils.printVector(outputData));
      
      double net, truth;
      net = network.getNetworkOutput()[0];
      truth = outputData[0];
      squaredError += Math.pow((net - truth), 2);
      System.out.println("Sum of Squared Errors: " + squaredError + "\n");
    }
  }
  
  
  

}
