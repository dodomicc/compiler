package Leetcode23;
import java.util.ArrayList;

public class Leetcode23Test {
    public static ListNode generateIncreasingSequence(int n, int maxIncrecement){
        if(n==0){
            ListNode head=new ListNode();
            return head;
        }else{
            ListNode head=new ListNode(Math.round((float)Math.random()*maxIncrecement));
            ListNode iterator=head;
            for(int i=1; i<n; i++){
                iterator.next=new ListNode(iterator.val+Math.round((float)Math.random()*maxIncrecement));
                iterator=iterator.next;
            }
            return head;
        }
    }




    public static ListNode[] generateKIncreasingSequence(int maxNumberOfIncreasingSequence, int maxSequenceLength,int maxIncrecement){
        ArrayList<ListNode> headArray=new ArrayList<>();
        int n=Math.round((float)Math.random()*maxNumberOfIncreasingSequence);
        for(int i=0; i<n; i++){
            int sequenceLength=Math.round((float)Math.random()*maxSequenceLength);
            if(generateIncreasingSequence(sequenceLength, maxIncrecement).val!=null){
                headArray.add(generateIncreasingSequence(sequenceLength, maxIncrecement));
            }
        }
        ListNode[] lists=new ListNode[headArray.size()];
        n=headArray.size();
        for(int i=0; i<n; i++){
            lists[i]=headArray.get(i);
        }
        return lists;
    }




    public static void displayGeneratedSequence(ListNode[] headArray){
        if(headArray.length==0){
            System.out.println("There is no array to merge!");
            System.exit(-1);
        }
        for(int i=0; i<headArray.length; i++){
            ListNode head=headArray[i];
            System.out.println("The "+(i+1)+"th array");
            displaySingleSequence(head);
        }
    }




    public static void displaySingleSequence(ListNode head){
        if(head.val!=null){
            ListNode iterator=head;
            while(iterator.val!=null){
                System.out.print(iterator.val);
                iterator=iterator.next;
                if(iterator==null)
                    break;
                System.out.print(",");
            }
        System.out.println();
        }
    }




    public static void main(String[] args) {
        int maxNumberOfIncreasingSequence=10;
        int maxSequenceLength=100;
        int maxIncrecement=50;
        ListNode[] headArray=generateKIncreasingSequence(maxNumberOfIncreasingSequence, maxSequenceLength, maxIncrecement);
        displayGeneratedSequence(headArray);
        System.out.println("The Result of merging sorted arrays above");
        ListNode headArray2=Leetcode23Solution.mergeSortedArrays(headArray);
        displaySingleSequence(headArray2);
    }
}
