package Leetcode31;
import java.lang.Math;

public class Leetcode31Test {
    public static int[] generateIntArray(int minIntArrayLength, int maxIntArrayLength, int minIntElement, int maxIntElement){
        int arrayLength=Math.round((float)Math.random()*(maxIntArrayLength-minIntArrayLength)+minIntArrayLength);
        int[] intArray=new int[arrayLength];
        for(int i=0; i<arrayLength; i++){
            intArray[i]=Math.round((float)Math.random()*(maxIntElement-minIntElement)+minIntElement);
        }
        return intArray;
    }

    
    public static void displayNextPermutation(int[] nextPermutation){
        System.out.println("Next Permutation");
        System.out.print("[ ");
        for(int i=0; i<nextPermutation.length; i++){
            System.out.print(nextPermutation[i]+" ");
        }
        System.out.println("]");
    } 

   public static void main(String[] args) {
        int[] intArray=generateIntArray(10, 10, 0, 100);
        System.out.println("Original Integer Array");
        System.out.print("[ ");
        for(int i=0; i<intArray.length; i++){
            System.out.print(intArray[i]+" ");
        }
        System.out.println("]");
        System.out.print("[ ");
        for(int i=0; i<5000; i++){
            intArray=Leetcode31Solution.nextPermutation(intArray);
            displayNextPermutation(intArray);
        }
    }
}
