package other;

import java.util.LinkedList;
import java.util.List;
import java.util.Stack;


public class MinStack {
	Stack<Integer> stack = new Stack<Integer>();
	int min = Integer.MAX_VALUE;
	
	/** initialize your data structure here. */
	public MinStack() {
		
	}

	public void push(int x) {
		if(x <= min){
			stack.push(min);
			min = x;
		}
		stack.push(x);
	}

	public void pop() {
		if(stack.isEmpty())return;
		int temp = stack.pop();
		if(temp == min)
			min = stack.pop();
		
	}

	public int top() {
		return stack.peek();
	}

	public int getMin() {
		return min;
	}
}
