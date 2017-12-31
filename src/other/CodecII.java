package other;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;

import exercise.TreeNode;

public class CodecII {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if(root == null)return "null";
        StringBuffer str = new StringBuffer();
        Stack<TreeNode> stack = new Stack<TreeNode>();
        stack.add(root);
        while(!stack.isEmpty()){
        	root = stack.pop();
        	str.append(root.val).append(",");
        	if(root.right != null)stack.push(root.right);
        	if(root.left != null)stack.push(root.left);
        }
        return str.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
    	if(data.equals("null"))return null;
        String[] split = data.split(",");
        Queue<Integer> queue = new LinkedList<Integer>();
        for(int i = 0;i < split.length;i++){
        	queue.add(Integer.parseInt(split[i]));
        }
        return Tree(queue);
    }
    
    public TreeNode Tree(Queue<Integer> queue){
    	if(queue.isEmpty()){
    		return null;
    	}
    	TreeNode root = new TreeNode(queue.poll());
        Queue<Integer> small = new LinkedList<Integer>();
    	while(!queue.isEmpty() && queue.peek() < root.val){
    		small.add(queue.poll());
    	}
    	root.left = Tree(small);
    	root.right = Tree(queue);
    	return root;
    }
}
