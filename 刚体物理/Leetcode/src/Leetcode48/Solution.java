
package Leetcode48;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashMap;



class Solution {
    
    public int kInversePairs(int n, int k) {
        int[] oldDp=new int[k+1];
        int[] newDp=new int[k+1];
        int modulo=(int)Math.pow(10,9)+7;
        oldDp[0]=1;
        for(int i=1; i<n; i++){
            newDp[0]=1;
            for(int j=1; j<=k; j++){
                newDp[j]=(newDp[j-1]+oldDp[j])%modulo;
                newDp[j]=newDp[j]-((Math.max(0,j-i)>Math.max(0,j-1-i))?oldDp[Math.max(0,j-1-i)]:0);
                newDp[j]=newDp[j]>=0?newDp[j]:newDp[j]+modulo;
            }
            for(int j=1; j<=k; j++) oldDp[j]=newDp[j];
        }
        for(int i=0; i<=k; i++) System.out.print(newDp[i]+" ");
        return newDp[k];
    }

    public static void main(String[] args) {
        int n=16;
        int[] arr=new int[n+1];
        arr[0]=1;
        arr[1]=1;
        for(int i=2; i<=n; i++){
            for(int j=1; j<=i; j++){
                arr[i]+=arr[j-1]*arr[i-j];
            }
        }
        for(int i=0; i<=n; i++){
            System.out.print(arr[i]+" ");
        }

        
       

    }
}
