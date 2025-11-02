package Leetcode23;

public class Leetcode23Solution {
  public static ListNode mergeTwoArray(ListNode[] headForTwoArray){
    if(headForTwoArray.length==0){
      System.out.println("There is no array to merge!");
      System.exit(1);
    }
    if(headForTwoArray.length==1){
      return headForTwoArray[0];
    }
    ListNode iterator1=headForTwoArray[0];
    ListNode iterator2=headForTwoArray[1];
    ListNode iterator= new ListNode();
    ListNode sortedArrayHead=iterator;
    while(iterator1!=null && iterator2!=null){
      if(iterator1.val<iterator2.val){
        iterator.next=new ListNode(iterator1.val);
        iterator1=iterator1.next;
        iterator=iterator.next;
      }else{
        iterator.next=new ListNode(iterator2.val);
        iterator2=iterator2.next;
        iterator=iterator.next;
      }
    }
    if(iterator1==null){
      iterator.next=iterator2;
    }else{
      iterator.next=iterator1;
    }
    return sortedArrayHead.next;
  } 




  public static ListNode mergeSortedArrays(ListNode[] headArrays){
    if(headArrays.length==0){
      System.out.println("There is no array to merge!");
      System.exit(1);
    }
    if(headArrays.length==1){
      return headArrays[0];
    }
    if(headArrays.length==2){
      return mergeTwoArray(headArrays);
    }
    ListNode iterator1 = headArrays[0];
    ListNode iterator2 = headArrays[1];
    ListNode result;
    for(int i=2; i<headArrays.length; i++){
      ListNode[] headForTwoArray=new ListNode[2];
      headForTwoArray[0]=iterator1;
      headForTwoArray[1]=iterator2;
      result=mergeTwoArray(headForTwoArray);
      iterator1=result;
      iterator2=headArrays[i];
    }
    ListNode[] headForTwoArray=new ListNode[2];
    headForTwoArray[0]=iterator1;
    headForTwoArray[1]=iterator2;
    result=mergeTwoArray(headForTwoArray);
    return result;
  }
}
