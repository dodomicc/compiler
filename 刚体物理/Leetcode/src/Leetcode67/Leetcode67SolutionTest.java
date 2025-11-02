package Leetcode67;

public class Leetcode67SolutionTest {
    public static void main(String[] args) {
        String a=new String("1010010111110011100111");
        String b=new String("1011110001001010101");
        int i=a.length()-1;
        int j=b.length()-1;
        int carry=0;
        String result="";
        while(i>=0||j>=0||carry>0){
            if(i>=0){
                carry=carry+(a.charAt(i) - '0');
                i--;
            }if(j>=0){
                carry=carry+(b.charAt(j) - '0');
                j--;
            }
            result=(carry%2)+result;
            carry=carry/2;
        }
        System.out.println(result);
    }
    
}
