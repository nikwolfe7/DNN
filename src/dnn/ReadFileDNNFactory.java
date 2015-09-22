package dnn;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class ReadFileDNNFactory implements DNNFactory {
  
  private NeuralNetwork network;

  public ReadFileDNNFactory(String fileName) throws IOException {
    File f = new File(fileName);
    Scanner reader = new Scanner(f);
    List<String> lines = new ArrayList<>();
    while(reader.hasNextLine()) 
      lines.add(reader.nextLine());
    Layer[] layers = new Layer[lines.size()];
    for(int i = 0; i < layers.length; i++)
      layers[i] = Layer.instantiateFromString(lines.get(i));
    this.network = new NeuralNetwork(layers);
    reader.close();
    printReport();
  }
  
  private void printReport() {
    StringBuilder sb = new StringBuilder();
    Layer[] layers = network.getLayers();
    for (int i = 0; i < layers.length - 1; i++)
      sb.append("| Hidden Layer " + (i + 1) + ":\t" + layers[i].getNumNeurons() + " Neurons\n");
    System.out.print("\nDNN Factory" 
            + "\n------------------------------------------------------"
            + "\nBuilding neural network with the following dimensions:"
            + "\n| Number of Inputs:\t" + network.getNumInputs() 
            + "\n| Number of Outputs:\t" + network.getNumOutputs() 
            + "\n" + sb.toString());
  }

  @Override
  public NeuralNetwork getInitializedNeuralNetwork() {
    return network;
  }

  public static void main(String[] args) throws IOException {
    /* Initialize a new network... */
    double[] input = { 0.1, 0.4 };
    DNNFactory factory = new SimpleDNNFactory(input.length, 2, 5, 4, 4);
    NeuralNetwork network = factory.getInitializedNeuralNetwork();
    
    /* Feed an input through the network... */
    network.feedForwardFromInput(input);
   
    /* Output the result */
    System.out.println("Network input:\t" + DNNUtils.printVector(input));
    System.out.println("Network output:\t" + DNNUtils.printVector(network.getNetworkOutput()));
    
    /* Write that network to file... */
    network.writeNetworkToFile("test.dnn.txt");
    
    /* Read the same network from file... */
    factory = new ReadFileDNNFactory("test.dnn.txt");
    network = factory.getInitializedNeuralNetwork();
    
    /* Feed an input through the network... */
    network.feedForwardFromInput(input);
    
    /* Output the result */
    System.out.println("\nNetwork input:\t" + DNNUtils.printVector(input));
    System.out.println("Network output:\t" + DNNUtils.printVector(network.getNetworkOutput()));
    
    /* Write the network to file... */
    network.writeNetworkToFile("new.test.dnn.txt");
    
    /* Verify that the initial network and saved network are identical... */
    Scanner scn1 = new Scanner(new File("test.dnn.txt"));
    Scanner scn2 = new Scanner(new File("new.test.dnn.txt"));
    boolean identical = true;
    while(scn1.hasNextLine()) {
      if(!scn1.nextLine().equals(scn2.nextLine())) {
        System.out.println("Files are NOT identical!!!");
        identical = false;
      }
    }
    System.out.println("\nStored DNN files are identical!");
    scn1.close();
    scn2.close();
  }

}
