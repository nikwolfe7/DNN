package dnn;

public class TransferFunction {
  
  public static double weightedSum(double[] inputs, double[] weights) {
    double sum = 0;
    for(int i = 0; i < inputs.length; i++) {
      sum += (inputs[i] * weights[i]);
    }
    return sum;
  }
  
  public static void main(String[] args) {
    double[] inputs = { 1, 3, 4, 5, 2, 1, 2 };
    double[] weights = { 0.21, 0.39, 0.43, 0.85, 0.62, 0.34, 0.13 };
    System.out.println("Weighted Sum: " + TransferFunction.weightedSum(inputs, weights));
  }

}
