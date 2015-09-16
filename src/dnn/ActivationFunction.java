package dnn;

public class ActivationFunction {
  
  public static double sigmoid(double input) {
    return 1.0 / (1.0 + Math.exp(-input));
  }
  
  public static void main(String[] args) {
    double i = -5;
    while(i < 5) {
      System.out.println(i + "," + ActivationFunction.sigmoid(i));
      i += 0.1;
    }
  }

}
