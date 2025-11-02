package Leetcode22;

class Leetcode22Test{
    public static void main(String[] args) {
        int maxNumberOfParenthesis=5;
        for(int n=1; n<maxNumberOfParenthesis+1; n++){
            System.out.print("number of parenthesis: "+n+"    ");
            System.out.println("number of generated parenthesis: "+Leetcode22Solution.generateParenthesis(n).size());
            System.out.println(Leetcode22Solution.generateParenthesis(n));
        }
    }
}
