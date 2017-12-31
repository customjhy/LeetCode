package other;

import java.util.LinkedList;
import java.util.Queue;

public class MyStack {
	public Queue<Integer> stack = new LinkedList<Integer>();
	
    /** Initialize your data structure here. */
    public MyStack() {
        
    }
    
    /** Push element x onto stack. */
    public void push(int x) {
        stack.add(x);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
    	if(stack.isEmpty())return -1;
        Queue<Integer> temp = new LinkedList<Integer>();
        int index = 0;
        index = stack.poll();
        if(stack.isEmpty()){
        	return index;
        }
        do{
        	temp.add(index);
        	index = stack.poll();
        }while(!stack.isEmpty());
        stack = temp;
        return index;
    }
    
    /** Get the top element. */
    public int top() {
    	if(stack.isEmpty())return -1;
        Queue<Integer> temp = new LinkedList<Integer>();
        int index = 0;
        while(!stack.isEmpty()){
        	index = stack.poll();
        	temp.add(index);
        }
        stack = temp;
        return index;
    }
    
    /** Returns whether the stack is empty. */
    public boolean empty() {
        return stack.isEmpty();
    }
}

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack obj = new MyStack();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.top();
 * boolean param_4 = obj.empty();
 */
