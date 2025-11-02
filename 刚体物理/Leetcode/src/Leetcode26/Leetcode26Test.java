package Leetcode26;
import java.lang.Math;

public class Leetcode26Test {
    public static int[] generateIncreasingSequence(int maxSequenceLength, int maxIncrement){
        int sequenceLength=Math.round((float)Math.random()*maxSequenceLength);
        if(sequenceLength>0){
            int[] increasingSequence=new int[sequenceLength];
            increasingSequence[0]=Math.round((float)Math.random()*maxIncrement);
            for(int i=1; i<sequenceLength; i++){
                increasingSequence[i]=increasingSequence[i-1]+Math.round((float)Math.random()*maxIncrement);
            }
            return increasingSequence;
        }
        System.out.println("The length of the generated sequence is 0!");
        System.exit(-1);
        return new int[1];
       
    }




   




    public static void main(String[] args) {
        int maxSequenceLength=10;
        int maxIncrement=1;
        int[] increasingSequence=generateIncreasingSequence(maxSequenceLength,maxIncrement);
        System.out.println("Original Sequence:");
        System.out.print("[ ");
        for (int i=0; i<increasingSequence.length; i++){
            System.out.print(increasingSequence[i]+" ");
        
        }
        System.out.println("]");
        System.out.println("Length of original sequence: "+increasingSequence.length);
        System.out.println("");
         System.out.println("Nonduplicated Sequence:");
        int LengthOfNonDuplicatedSequence=Leetcode26Solution.removeDuplicateElement(increasingSequence);
        System.out.print("Length of nonDuplicated sequence: ");

        System.out.println(LengthOfNonDuplicatedSequence);
    }
}
