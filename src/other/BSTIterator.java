package other;

import java.util.ArrayList;
import java.util.List;

import javax.swing.RootPaneContainer;

import exercise.TreeNode;

/**
 * Definition for binary tree
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */

public class BSTIterator {
	List<Integer> list = new ArrayList<>();
	int i = 0;

	public BSTIterator(TreeNode root) {
		preIterator(root);
	}

    public void preIterator(TreeNode root){
        if(root == null)return;
        preIterator(root.left);
        list.add(root.val);
        preIterator(root.right);
    }
    
    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return i < list.size();
    }

    /** @return the next smallest number */
    public int next() {
        return list.get(i++);
    }
}

/**
 * Your BSTIterator will be called like this:
 * BSTIterator i = new BSTIterator(root);
 * while (i.hasNext()) v[f()] = i.next();
 */
