package Leetcode26;

import java.util.HashMap;

public class Leetcode26Solution {
   public static int removeDuplicateElement(int[] increasingSequence){
       int nonDuplicateElement=0;
       HashMap<Integer,Integer> map=new HashMap<>();
       System.out.print("[ ");
       for(int i=0; i<increasingSequence.length; i++){
           if(!map.containsKey(increasingSequence[i])){
               map.put(increasingSequence[i],i);
               System.out.print(increasingSequence[i]+" ");
               nonDuplicateElement++;
           }
               
       }
       System.out.println("]");
      
       return nonDuplicateElement;
   }
}
