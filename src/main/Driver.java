package main;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

import training.DataGenerator;
import training.DataInstance;
import dnn.DNNFactory;
import dnn.DNNUtils;
import dnn.NeuralNetwork;
import dnn.SimpleDNNFactory;

public class Driver {
  
  public static void main(String[] args) throws FileNotFoundException, IOException {
    /* Digit summation ... */
    int numInputs = 26;
    int numDataInstances = 5;
    int numOutputs = 1;
    String dataFile = DataGenerator.getAdditionData(numInputs, numDataInstances);
    List<DataInstance> data = DNNUtils.getTrainingInstances(dataFile, numInputs, numOutputs);
    DNNFactory factory = new SimpleDNNFactory(numInputs, numOutputs, 250, 25, 250, 25, 300, 60, 250, 25);
    trainDNN(data, factory.getInitializedNeuralNetwork());
    
    /* Cosine */
    dataFile = DataGenerator.getCosineData(1000);
    data = DNNUtils.getTrainingInstances(dataFile, 1, 1); // 1 in, 1 out
    trainDNN(data, factory.getInitializedNeuralNetwork());
    
    /* Pythagorean */
    dataFile = DataGenerator.getPythagoreanData(1000);
    data = DNNUtils.getTrainingInstances(dataFile, 2, 1); // 2 in, 1 out
    trainDNN(data, factory.getInitializedNeuralNetwork());
  }
  
  private static void trainDNN(List<DataInstance> data, NeuralNetwork network) {
    double squaredError = 0;
    for (DataInstance instance : data) {

      /* Run the data through the network */
      network.feedForwardFromInput(instance.getInputVector());
      network.backPropagateError(instance.getOutputTruthValue());

      /* Output */
      System.out.println("Input:\t" + DNNUtils.printVector(instance.getInputVector()));
      System.out.println("Output:\t" + DNNUtils.printVector(network.getNetworkOutput()));
      System.out.println("Truth:\t" + DNNUtils.printVector(instance.getOutputTruthValue()));

      /* Running sum of the error */
      double net, truth;
      net = network.getNetworkOutput()[0];
      truth = instance.getOutputTruthValue()[0];
      squaredError += Math.pow((net - truth), 2);
      System.out.println("Sum of Squared Errors: " + squaredError + "\n");
    }
  }
  
  
  

}
