package Leetcode31;

import java.util.Arrays;


public class Leetcode31Solution {
    public static int[] nextPermutation(int[] intArray){
        int flag=0;
        int iterator1=intArray.length-2;
        int[] changePairs=new int[2];
        changePairs[1]=Arrays.stream(intArray).max().getAsInt();
        while(flag==0 && iterator1>=0){
            for(int iterator2=iterator1+1; iterator2<intArray.length; iterator2++){
                if(intArray[iterator2]>intArray[iterator1] && intArray[iterator2]<=changePairs[1]){
                    flag=1;
                    changePairs[0]=iterator2;
                    changePairs[1]=intArray[iterator2];
                }
            }
            if(flag==1){
                int temp=intArray[iterator1];
                intArray[iterator1]=changePairs[1];
                intArray[changePairs[0]]=temp;   
                int[] sortedArray=Arrays.copyOfRange(intArray, iterator1+1, intArray.length);
                Arrays.sort(sortedArray);
                for(int i=iterator1+1; i<intArray.length; i++){
                    intArray[i]=sortedArray[i-(iterator1+1)];
                }
            }
            iterator1--;
        }
        return intArray;
    }
}
