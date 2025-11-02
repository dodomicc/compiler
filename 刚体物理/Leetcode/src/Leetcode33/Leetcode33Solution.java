package Leetcode33;

import java.util.Arrays;

public class Leetcode33Solution {
    
    public static int binarySearch(int [] targetArray, int target){
        int left=0;
        int right=targetArray.length-1;
        if(targetArray[left]>target || targetArray[right]<target)
            return -1;
        if(targetArray[left]==target){
            return left;
        }else if(targetArray[right]==target){
            return right;
        }else{
            while(right-left>1){
                int middle=(left+right)/2;
                if(target==targetArray[middle]){
                    return middle;
                }else if(target<targetArray[middle]){
                    right=middle;
                }else{
                    left=middle;
                }
            }
            return -1;
        }
    }

    public static boolean isRotated(int[] targetArray){
        if(targetArray[0]<targetArray[targetArray.length-1]){
            return false;
        }
        return true;
    }


    public static int findSplitPoint(int[] targetArray){
        int left=0;
        int right=targetArray.length-1;
        if(targetArray[left]>targetArray[left+1]){
            return left;
        }else if(targetArray[right]<targetArray[right-1]){
            return right-1;
        }else{
            while(right-left>1){
                int middle=(left+right)/2;
                if(targetArray[middle]>targetArray[middle+1]){
                    return middle;
                }else if(targetArray[middle]<targetArray[left]){
                    right=middle;
                }else{
                    left=middle;
                }
            }
            if(targetArray[left]>targetArray[left+1]){
                return left;
            }else{
                return right;
            }
        }
    }

    public static int search(int[] targetArray, int target){
        if(isRotated(targetArray)){
            int temp=findSplitPoint(targetArray);
            int[] targetArray1=Arrays.copyOfRange(targetArray, 0, temp+1);
            int[] targetArray2=Arrays.copyOfRange(targetArray, temp+1, targetArray.length);
            int result1=binarySearch(targetArray1, target);
            int result2=binarySearch(targetArray2, target);
            if(result1*result2>0){
                return -1;
            }else{
                if(result1>=0){
                    return result1;
                }else{
                    return result2+temp+1;
                }
            }
        }else{
            return binarySearch(targetArray, target);
        }
    }
}

