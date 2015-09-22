package training;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DataGenerator {

  public static String getAdditionData(int numInputs, int sizeDataset) throws IOException {
    String fileName = "add-data.txt";
    List<double[]> data = getData(numInputs, sizeDataset, -100, 100);
    return applyFunctionAndWriteDataToFile(fileName, data, new InputFunction() {
      @Override
      public double[] calc(double... inputs) {
        double[] result = new double[1];
        for (double d : inputs)
          result[0] += d;
        return result;
      }
    });
  }
  
  public static String getPythagoreanData(int sizeDataset) throws IOException {
    String fileName = "pythagorean-data.txt";
    List<double[]> data = getData(2, sizeDataset, -1000, 1000);
    return applyFunctionAndWriteDataToFile(fileName, data, new InputFunction() {
      @Override
      public double[] calc(double... inputs) {
        double[] result = new double[1];
        result[0] = Math.sqrt(Math.pow(inputs[0], 2) + Math.pow(inputs[1], 2));
        return result;
      }
    });
  }
  
  public static String getCosineData(int sizeDataset) throws IOException {
    String fileName = "cos-data.txt";
    List<double[]> data = getData(1, sizeDataset, -360, 360);
    return applyFunctionAndWriteDataToFile(fileName, data, new InputFunction() {
      @Override
      public double[] calc(double... inputs) {
        double[] result = new double[1];
        result[0] = Math.cos(Math.PI * inputs[0] / 180.0);
        return result;
      }
    });
  }
  
  private static String applyFunctionAndWriteDataToFile(String fileName, List<double[]> data,
          InputFunction function) throws IOException {
    File file = new File(fileName);
    FileWriter writer = new FileWriter(file);
    for (double[] arr : data) {
      double[] result = function.calc(arr);
      String[] strArr = new String[arr.length + result.length];
      for (int i = 0; i < strArr.length; i++) {
        if (i < result.length)
          strArr[i] = "" + result[i];
        else
          strArr[i] = "" + arr[i - result.length];
      }
      writer.write(String.join(",", strArr) + "\n");
    }
    writer.close();
    return file.getCanonicalPath();
  }

  private static List<double[]> getData(int numInputs, int sizeDataset, double low, double high) {
    List<double[]> data = new ArrayList<double[]>();
    Random rnd = new Random();
    for (int i = 0; i < sizeDataset; i++) {
      double[] inputs = new double[numInputs];
      for (int j = 0; j < inputs.length; j++) {
        inputs[j] = rnd.nextDouble() * (high - low) + low;
      }
      data.add(inputs);
    }
    return data;
  }

}
