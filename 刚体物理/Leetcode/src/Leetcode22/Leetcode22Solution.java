package Leetcode22;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Leetcode22Solution {
    public static List<String>generateParenthesis(int n){
        List<String> Parenthesis = new ArrayList<String>();
        if(n==0){
           Parenthesis.add("");
        }else if(n==1){
           Parenthesis.add("()");
        }else{
            for(int left=1; left<n; left++){
                List<String>leftParenthesis=generateParenthesis(left);
                List<String>rightParenthesis=generateParenthesis(n-left);
                for (int i=0; i<leftParenthesis.size(); i++){
                    for (int j=0; j<rightParenthesis.size(); j++){
                       Parenthesis.add(leftParenthesis.get(i)+rightParenthesis.get(j));
                    }
                }
            }
            List<String>NestedParenthesis=generateParenthesis(n-1);
            for (int i=0; i<NestedParenthesis.size(); i++){
                   Parenthesis.add("("+NestedParenthesis.get(i)+")");
                    }
            }
        Set<String> uniqueParenthesis=new HashSet<>(Parenthesis);
        return new ArrayList<String>(uniqueParenthesis);
    }
}
