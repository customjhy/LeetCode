package other;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
//432. All O`one Data Structure
class AllOne {
    class valueNode {
        valueNode preNode;             
        valueNode nextNode;           
        int value;                     // curNode.value;
        List<String> curKeys;         // store the key at the value of curNode.value;
        
        valueNode(int value,String key) {
            this.value = value;
            curKeys = new LinkedList<String>();
            curKeys.add(key);
        }
    }
	
    valueNode head = null;
    valueNode tail = null;
    valueNode curNode = null;
    Map<String, valueNode> map = new HashMap<>();
    
    
    /** Initialize your data structure here. */
    public AllOne() {
        
    }
    
    /** Inserts a new key <Key> with value 1. Or increments an existing key by 1. */
    public void inc(String key) {
        if(tail == null){
        	curNode = new valueNode(1, key);
        	head = curNode;
        	tail = curNode;
        	map.put(key, curNode);
        }else if(!map.containsKey(key)){
        	if(tail.value == 1){
        		tail.curKeys.add(key);
        		map.put(key, tail);
        	}else{
        		curNode = new valueNode(1, key);
        		curNode.preNode = tail;
        		tail.nextNode = curNode;
        		tail = tail.nextNode;
        		map.put(key, curNode);
        	}
        }else{
        	curNode = map.get(key);
        	if(curNode.preNode != null){
        		if(curNode.value + 1 == curNode.preNode.value){
        			curNode.curKeys.remove(key);
        			curNode.preNode.curKeys.add(key);
        			map.put(key, curNode.preNode);
        			checkEmpty(curNode);
        		}else{
        			valueNode newNode = new valueNode(curNode.value + 1, key);
                    newNode.preNode = curNode.preNode;
                    newNode.nextNode= curNode;
                    newNode.preNode.nextNode = newNode;
                    curNode.preNode = newNode;
                    curNode.curKeys.remove(key); 
                    map.put(key,newNode);
                    checkEmpty(curNode);
        		}
            }else {//which means the node is the head. so we build a new head.
                head = new valueNode(curNode.value+1,key);
                head.nextNode = curNode;
                curNode.preNode = head;
                curNode.curKeys.remove(key);
                map.put(key,head);
                checkEmpty(curNode);
            }
        }
    }
    
    /** Decrements an existing key by 1. If Key's value is 1, remove it from the data structure. */
    public void dec(String key) {
        if (head ==null ||!map.containsKey(key)) return; //which means nothing here.
                                                     //or  means no key in the structrue.
        curNode = map.get(key);
        if (curNode.nextNode != null) {  //which means the node is in the middle.
            if (curNode.nextNode.value == curNode.value - 1){ //which means we can just 
                curNode.nextNode.curKeys.add(key);
                curNode.curKeys.remove(key); 
                map.put(key,curNode.nextNode);
                checkEmpty(curNode);
            }else {                     //which means the nextNode value != curNode.value-1;
                    valueNode newNode = new valueNode(curNode.value-1, key);
                    newNode.nextNode = curNode.nextNode;
                    newNode.preNode= curNode;
                    newNode.nextNode.preNode = newNode;
                    curNode.curKeys.remove(key); 
                    curNode.nextNode = newNode;
                    map.put(key,newNode);
                    checkEmpty(curNode);
            }
        } else {    //which means the node is the tail. so we build a new head.
            if (curNode.value == 1) {     //just to delete the key.
                curNode.curKeys.remove(key);  
                map.remove(key);
                checkEmpty(curNode);
            }else {                         // build another tail.
                tail = new valueNode(curNode.value-1,key);
                tail.preNode = curNode;
                curNode.nextNode = tail;
                curNode.curKeys.remove(key);  
                map.put(key,tail);
                checkEmpty(curNode);
            }
        }
    }
    
    /** Returns one of the keys with maximal value. */
    public String getMaxKey() {
        if(head == null)return "";
        return head.curKeys.get(0);
    }
    
    /** Returns one of the keys with Minimal value. */
    public String getMinKey() {
        if(tail == null)return "";
        return tail.curKeys.get(0);
    }
    
    // to check whether the node should be delete because the keyList is empty.
    private void checkEmpty(valueNode checkNode) {
            if (checkNode.curKeys.size() != 0) return;
            if (checkNode.preNode == null && checkNode.nextNode == null){
                tail = null;
                head = null;
            }else if (checkNode.preNode == null && checkNode.nextNode != null) {
                head = checkNode.nextNode;
                head.preNode = null;
            }else if (checkNode.nextNode == null && checkNode.preNode != null){
                tail = checkNode.preNode;
                tail.nextNode = null;
            }else {
                checkNode.preNode.nextNode = checkNode.nextNode;
                checkNode.nextNode.preNode  = checkNode.preNode; 
            } 
    }
}

/**
 * Your AllOne object will be instantiated and called as such:
 * AllOne obj = new AllOne();
 * obj.inc(key);
 * obj.dec(key);
 * String param_3 = obj.getMaxKey();
 * String param_4 = obj.getMinKey();
 */
