/**
 * @author jhy
 * code from 12.31 to 1.20
 * 33 questions
 */

/**
 * 2017/12/31
 * 688. Knight Probability in Chessboard
 * 
class Solution {
	int[][] moves = {{1,-2},{1,2},{2,-1},{2,1},{-2,1},{-2,-1},{-1,-2},{-1,2}};
    public double knightProbability(int N, int K, int r, int c) {
        double[][] dp0 = new double[N][N];
        for(double[] row : dp0){Arrays.fill(row, 1);}
        for(int k = 0;k < K;k++){
        	double[][] dp1 = new double[N][N];
        	for(int i = 0;i < N;i++){
        		for(int j = 0;j < N;j++){
        			for(int[] move : moves){
        				int row = move[0] + i;
        				int col = move[1] + j;
        				if(isLegal(N, row, col))dp1[i][j] += dp0[row][col];
        			}
        		}
        	}
        	dp0 = dp1;
        }
        return dp0[r][c] / Math.pow(8, K);
    }
    
    public boolean isLegal(int N,int row,int col){
    	return row >= 0 && row < N && col >= 0 && col < N;
    }
}
 */

/**
 * 2018/1/1
 * 474. Ones and Zeroes
 * 
class Solution {
    public int findMaxForm(String[] strs, int m, int n) {
        int[][] max = new int[m + 1][n + 1];
        int[] num;
        for(String str : strs){
        	num = count01(str);
        	for(int i = m;i >= num[0];i--){
        		for(int j = n;j >= num[1];j--){
        			max[i][j] = Math.max(max[i][j], max[i - num[0]][j - num[1]] + 1);
        		}
        	}
        }
        return max[m][n];
    }
    
    public int[] count01(String str){
    	int[] count = new int[2];
    	for(int i = 0;i < str.length();i++){
    		count[str.charAt(i) - '0']++;
    	}
    	return count;
    }
}

 * 2018/1/1
 * 416. Partition Equal Subset Sum
 * 
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for(int num : nums){
        	sum += num;
        }
        if(sum % 2 == 1)return false;
        sum = sum / 2;
        boolean[][] dp = new boolean[nums.length + 1][sum + 1];
        for(int i = 0;i < nums.length + 1;i++)dp[i][0] = true;
        for(int i = 1;i <= nums.length;i++){
        	for(int j = 1;j <= sum;j++){
        		dp[i][j] = dp[i - 1][j];
        		if(j >= nums[i - 1]){
        			dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]];
        		}
        	}
        }
        return dp[nums.length][sum];
    }
    
 * 2018/1/1
 * 75. Sort Colors
 * 
class Solution {
    public void sortColors(int[] nums) {
        if(nums.length == 0)return;
        int zero = 0,two = nums.length - 1;
        for(int i = 0;i <= two;i++){
        	while(nums[i] == 2 && i < two)swap(nums, i, two--);
        	while(nums[i] == 0 && i > zero)swap(nums, i, zero++);
        }
    }
    
    public void swap(int[] nums,int a,int b){
    	int temp = nums[a];
    	nums[a] = nums[b];
    	nums[b] = temp;
    }
}

 * 2018/1/1
 * 670. Maximum Swap
 * 
class Solution {
    public int maximumSwap(int num) {
    	if(num <= 10)return num;
        char[] nums = Integer.toString(num).toCharArray();
        for(int i = 0;i < nums.length - 1;i++){
        	int index = getBigger(nums, i);
        	if(index != -1){
        		char temp = nums[i];
        		nums[i] = nums[index];
        		nums[index] = temp;
        		return Integer.parseInt(new String(nums));
        	}
        }
        return num;
    }
    
    public int getBigger(char[] nums,int start){
    	if(nums[start] == '9')return -1;
    	char max = nums[start];
    	int index = start;
    	for(int i = nums.length - 1;i > start;i--){
    		if(nums[i] > max){
    			max = nums[i];
    			index = i;
    		}
    	}
    	if(index == start)return -1;
    	else return index;
    }
}

 * 2018/1/1
 * 24. Swap Nodes in Pairs
 * 
    public ListNode swapPairs(ListNode head) {
    	if(head == null || head.next == null)return head;
    	ListNode temp = head;
    	head = temp.next;
        ListNode pre = temp.next;
    	temp.next = swapPairs(temp.next.next);
    	pre.next = temp;
    	return head;
    }
    
 * 2018/1/1
 * 162. Find Peak Element
 * 
class Solution {
    public int findPeakElement(int[] nums) {
        return helper(nums, 0, nums.length - 1);
    }
    
    public int helper(int[] nums,int start,int end){
    	if(start == end)return start;
    	if(start + 1 == end){
    		if(nums[end] > nums[start])return end;
    		return start;
    	}
    	int mid = (start + end) / 2;
    	if(nums[mid] > nums[mid - 1] && nums[mid] > nums[mid + 1])return mid;
    	else if(nums[mid] > nums[mid - 1] && nums[mid] < nums[mid + 1])return helper(nums, mid + 1, end);
    	else return helper(nums, start, mid - 1);
    }
}

 * 2018/1/1
 * 90. Subsets II
 * 
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Set<List<Integer>> set = new HashSet<>();
        set.add(new ArrayList<Integer>());
        for(int num : nums){
        	Set<List<Integer>> temp = new HashSet<>();
        	for(List<Integer> list : set){
        		temp.add(list);
        		List<Integer> newList = new ArrayList<>(list);
        		newList.add(num);
        		Collections.sort(newList);
        		temp.add(newList);
        	}
        	set = temp;
        }
        List<List<Integer>> res = new ArrayList<>();
        for(List<Integer> list : set){
        	res.add(list);
        }
        return res;
    }
 */

/**
 * 2018/1/2
 * 698. Partition to K Equal Sum Subsets
 * 
class Solution {
    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = 0;
    	for(int num : nums)sum += num;
    	if(k <= 0 || sum % k != 0)return false;
    	int target = sum / k;
    	boolean[] isVisited = new boolean[nums.length];
    	return partitionKSubsets(nums, isVisited, 0, k, 0, target);
    }
    
    public boolean partitionKSubsets(int[] nums, boolean[] isVisited, int start,int k, int sum, int target){
    	if(k == 1)return true;
    	if(sum == target)return partitionKSubsets(nums, isVisited, 0, k - 1, 0, target);
    	for(int i = start;i < isVisited.length;i++){
    		if(!isVisited[i]){
    			isVisited[i] = true;
    			if(partitionKSubsets(nums, isVisited, i +  1, k, sum + nums[i], target))return true;
    			isVisited[i] = false;
    		}
    	}
    	return false;
    }
}
 */

/**
 * 2018/1/3
 * 129. Sum Root to Leaf Numbers
 * 
class Solution {
	public int sumNumbers(TreeNode root) {
		return sum(root, 0);
	}

	public int sum(TreeNode n, int s) {
		if (n == null)
			return 0;
		if (n.right == null && n.left == null)
			return s * 10 + n.val;
		return sum(n.left, s * 10 + n.val) + sum(n.right, s * 10 + n.val);
	}
};
 */

/**
 * 2018/1/4
 * 313. Super Ugly Number
 * 
    public int nthSuperUglyNumber(int n, int[] primes) {
        int[] ugly = new int[n];
        int[] index = new int[primes.length];
        ugly[0] = 1;
        for(int i = 1;i < ugly.length;i++){
        	ugly[i] = Integer.MAX_VALUE;
        	for(int j = 0;j < primes.length;j++){
        		ugly[i] = Math.min(ugly[i], primes[j] * ugly[index[j]]);
        	}
        	for(int j = 0;j < primes.length;j++){
        		while(primes[j] * ugly[index[j]] <= ugly[i])index[j]++;
        	}
        }
        return ugly[n - 1];
    }
    
 * 2018/1/4
 * 264. Ugly Number II
 * 
    public int nthUglyNumber(int n) {
        int[] element = {2,3,5};
        int[] ugly = new int[n];
        int[] index = new int[3];
        ugly[0] = 1;
        for(int i = 1;i < n;i++){
        	ugly[i] = Integer.MAX_VALUE;
        	for(int j = 0;j < 3;j++){
        		ugly[i] = Math.min(ugly[i], element[j] * ugly[index[j]]);
        	}
        	for(int j = 0;j < 3;j++){
        		while(element[j] * ugly[index[j]] <= ugly[i])index[j]++;
        	}
        }
        return ugly[n - 1];
    }
    
 * 2018/1/4
 * 662. Maximum Width of Binary Tree
 * 
    public int widthOfBinaryTree(TreeNode root) {
        if(root == null)return 0;
    	int max = 1;
    	Queue<TreeNode> queue = new LinkedList<>();
    	Queue<Integer> num = new LinkedList<>();
    	queue.add(root);
    	TreeNode temp = root;
    	num.add(1);
    	while(!queue.isEmpty()){
    		int size = queue.size();
    		temp = queue.poll();
    		int left = num.poll();
    		if(temp.left != null){
    			queue.add(temp.left);
    			num.add(2 * left);
    		}
    		if(temp.right != null){
    			queue.add(temp.right);
    			num.add(2 * left + 1);
    		}
    		for(int i = 1;i < size;i++){
    			temp = queue.poll();
    			int right = num.poll();
    			max = Math.max(max, right - left + 1);
        		if(temp.left != null){
        			queue.add(temp.left);
        			num.add(2 * right);
        		}
        		if(temp.right != null){
        			queue.add(temp.right);
        			num.add(2 * right + 1);
        		}
    		}
    	}
    	return max;
    }
 */
    
/**
 * 2018/1/5
 * 735. Asteroid Collision
 * 
    public int[] asteroidCollision(int[] asteroids) {
        List<Integer> list = new ArrayList<>();
        Stack<Integer> stack = new Stack<>();
        int index = 0;
        for(index = 0;index < asteroids.length;index++){
        	if(asteroids[index] < 0)list.add(asteroids[index]);
        	else break;
        }
        if(index == asteroids.length){//如果输入全部为负数
        	int[] res = new int[list.size()];
        	for(int i = 0;i < list.size();i++)res[i] = list.get(i);
        	return res;
        }
        for(int i = index;i < asteroids.length;i++){
        	if(asteroids[i] > 0)stack.add(asteroids[i]);
        	else{
        		boolean flag = true;//为true，则负数行星为栈中最大，在list中添加
        		while(!stack.isEmpty() && flag){
        			if(stack.peek() < -asteroids[i]){
        				stack.pop();
        			}
        			else if(stack.peek() == -asteroids[i]){
        				stack.pop();
        				flag = false;
        			}
        			else{
        				flag = false;
        			}
        		}
        		if(flag == true)list.add(asteroids[i]);
        	}
        }
        Stack<Integer> temp = new Stack<>();//将stack中数据转为正序
        while(!stack.isEmpty())temp.add(stack.pop());
        while(!temp.isEmpty())list.add(temp.pop());
        int[] res = new int[list.size()];
        for(int i = 0;i < list.size();i++)res[i] = list.get(i);
        return res;
    }
    
 * 2018/1/5
 * 49. Group Anagrams
 * 
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> res = new ArrayList<>();
        Map<String, Integer> map = new HashMap<>();
        int count = 0;
        for(String str : strs){
        	String hash = fakeHash(str);
        	if(map.containsKey(hash)){
        		res.get(map.get(hash)).add(str);
        	}
        	else{
        		map.put(hash, count);
        		count++;
        		List<String> list = new ArrayList<>();
        		list.add(str);
        		res.add(list);
        	}
        }
        return res;
    }
    
    String fakeHash(String str){
    	int[] cha = new int[26];
    	for(char c : str.toCharArray())cha[c - 'a']++;
    	StringBuffer res = new StringBuffer();
    	for(int i = 0;i < 26;i++){
    		if(cha[i] != 0){
    			res.append(cha[i]).append((char)(i + 'a'));
    		}
    	}
    	return res.toString();
    }
}

 * 2018/1/5
 * 640. Solve the Equation
 * 
class Solution {
    public String solveEquation(String equation) {
        String[] equas = equation.split("=");
        int[] left = parameter(equas[0]);
        int[] right = parameter(equas[1]);
        if(left[0] == right[0] && left[1] == right[1])return "Infinite solutions";
        else if(left[0] == right[0] && left[1] != right[1])return "No solution";
        int res = (right[1] - left[1]) / (left[0] - right[0]);
        return "x=" + res;
    }
    
    public int[] parameter(String equa){
    	int[] para = new int[2];
    	int i = 0;
    	int left = 0;
    	if(equa.charAt(i) == '-' || equa.charAt(i) == '+'){
    		i++;
    	}
    	while(i != equa.length()){
    		while(i != equa.length() && equa.charAt(i) != '+' && equa.charAt(i) != '-')i++;
    		if(equa.charAt(i - 1) == 'x'){
    			if(i - 1 == 0 || equa.charAt(i - 2) == '+')para[0]++;
    			else if(i - 1 == 0 || equa.charAt(i - 2) == '-')para[0]--;
    			else para[0] += Integer.parseInt(equa.substring(left, i - 1));
    		}
    		else{
    			para[1] += Integer.parseInt(equa.substring(left, i));
    		}
    		if(i == equa.length())return para;
    		left = i++;
    	}
    	return para;
    }
}
 */

/**
 * 2018/1/6
 * 279. Perfect Squares
 * 
	public int numSquares(int n) {
		if(n <= 0)return 0;
		int[] arr = new int[n + 1];
		Arrays.fill(arr, Integer.MAX_VALUE);
		arr[0] = 0;
		for (int i = 0; i <= n; i++) {
			for(int j = 1;j * j <= i;j++){
				arr[i] = Math.min(arr[i], arr[i - j * j] + 1);
			}
		}
		return arr[n];
	}
 */

/**
 * 2018/1/7
 * 450. Delete Node in a BST
 * 
class Solution {
	TreeNode father = new TreeNode(0);
	boolean isLeft = true;
	
    public TreeNode deleteNode(TreeNode root, int key) {
        TreeNode delNode = search(root, key);
        if(delNode == null)return root;
        if(delNode.left == null && delNode.right == null){
            if(delNode == root)return null;
        	if(isLeft){
        		father.left = null;
        	}
        	else{
        		father.right = null;
        	}
        }
        else if(delNode.left == null && delNode.right != null){
        	delNode.val = delNode.right.val;
        	delNode.left = delNode.right.left;
        	delNode.right = delNode.right.right;
        }
        else if(delNode.left != null && delNode.right == null){
        	delNode.val = delNode.left.val;
        	delNode.right = delNode.left.right;
        	delNode.left = delNode.left.left;
        }
        else{//删除节点左右子节点均不为空
        	TreeNode rightNode = delNode.right;
        	if(rightNode.left == null){
        		delNode.val = rightNode.val;
        		delNode.right = rightNode.right;
        	}
        	else{
        		TreeNode leftNode = rightNode.left;
        		TreeNode pre = rightNode;
        		while(leftNode.left != null){
        			leftNode = leftNode.left;
        			pre = pre.left;
        		}
        		pre.left = leftNode.right;
        		delNode.val = leftNode.val;
        	}
        }
        return root;
    }
    
    public TreeNode search(TreeNode root,int key){
    	if(root == null)return null;
    	if(root.val == key){
    		father.left = root;
    		isLeft = true;
    		return root;
    	}
    	else if(root.val < key){
    		if(root.right != null && root.right.val == key){
    			father = root;
    			isLeft = false;
    			return root.right;
    		}
    		return search(root.right, key);
    	}
    	else {
    		if(root.left != null && root.left.val == key){
    			father = root;
    			isLeft = true;
    			return root.left;
    		}
    		return search(root.left, key);
    	}
    }
}

 * 2018/1/7
 * 760. Find Anagram Mappings
 * 
    public int[] anagramMappings(int[] A, int[] B) {
        int[] res = new int[A.length];
        for(int i = 0;i < A.length;i++){
        	for(int j = 0;j < B.length;j++){
        		if(A[i] == B[j]){
        			res[i] = j;
        			break;
        		}
        	}
        }
        return res;
    }
 */

/**
 * 2018/1/8
 * 289. Game of Life
 * 
class Solution {
    public void gameOfLife(int[][] board) {
    	if(board.length == 0 || board[0].length == 0)return;
        int row = board.length;
        int column = board[0].length;
    	int[][] res = new int[row][column];
    	for(int i = 0;i < row;i++){
    		for(int j = 0;j < column;j++){//计算res的值
    			int count = 0;
    			for(int m : new int[]{-1,0,1}){
    				for(int n : new int[]{-1,0,1}){
    					if(isValid(i + m, j + n, row, column)){
    						if(board[i + m][j + n] == 1)count++;
    					}
    				}
    			}
    			if(board[i][j] == 1)count--;
    			if(board[i][j] == 1){
    				if(count < 2 || count > 3)res[i][j] = 0;
    				else res[i][j] = 1;
    			}
    			else{
    				if(count == 3)res[i][j] = 1;
    				else res[i][j] = 0;
    			}
    		}
    	}
    	for(int i = 0;i < row;i++){
    		for(int j = 0;j < column;j++){
    			board[i][j] = res[i][j];
    		}
    	}
    }
    
    public boolean isValid(int i,int j,int row,int column){
    	return i >= 0 && j >= 0 && i < row && j < column;
    }
}

 * 2018/1/8
 * 388. Longest Absolute File Path
 * 
    public int lengthLongestPath(String input) {
        int res = 0;
        Map<Integer, Integer> m = new HashMap<>();
        m.put(0, 0);
        for (String s : input.split("\n")) {
            int level = s.lastIndexOf("\t") + 1;
            int len = s.substring(level).length();
            if (s.contains(".")) {
                res = Math.max(res, m.get(level) + len);
            } else {
                m.put(level + 1, m.get(level) + len + 1);
            }
        }
        return res;
    }
    
 * 2018/1/8
 * 36. Valid Sudoku
 * 
    public boolean isValidSudoku(char[][] board) {
    	if(board.length != 9 || board[0].length != 9)return false;
        Set<Character> set = new HashSet<>();
        for(int i = 0;i < 9;i++){
        	set.clear();
        	for(int j = 0;j < 9;j++){
        		if(board[i][j] != '.'){
        			if(set.contains(board[i][j]))return false;
        			set.add(board[i][j]);
        		}
        	}
        }
        for(int j = 0;j < 9;j++){
        	set.clear();
        	for(int i = 0;i < 9;i++){
        		if(board[i][j] != '.'){
        			if(set.contains(board[i][j]))return false;
        			set.add(board[i][j]);
        		}
        	}
        }
    	for(int i : new int[]{0,3,6}){
    		for(int j : new int[]{0,3,6}){
    	    	set.clear();
    	    	for(int m : new int[]{0,1,2}){
    	    		for(int n : new int[]{0,1,2}){
    	    			if(board[i + m][j + n] != '.'){
    	    				if(set.contains(board[i + m][j + n]))return false;
    	    				set.add(board[i + m][j + n]);
    	    			}
    	    		}
    	    	}
    		}
    	}
    	return true;
    }
    
 * 2018/1/8
 * 116. Populating Next Right Pointers in Each Node
 * 
public class Solution {
	public class TreeLinkNode {
		int val;
		TreeLinkNode left, right, next;
		TreeLinkNode(int x) { val = x; }
	}
    public void connect(TreeLinkNode root) {
        if(root == null)return;
        TreeLinkNode cur = null;
        TreeLinkNode pre = root;
        while(pre.left != null){
        	cur = pre;
        	cur.left.next = cur.right;
        	while(cur.next != null){
        		cur.right.next = cur.next.left;
        		cur = cur.next;
        		cur.left.next = cur.right;
        	}
        	pre = pre.left;
        }
    }
}

 * 2018/1/8
 * 11. Container With Most Water
 * 
    public int maxArea(int[] height) {
    	if(height.length < 2)return 0;
        int max = Integer.MIN_VALUE;
        int low = 0;
        int high = height.length - 1;
        while(low < high){
        	max = Math.max(max, (high - low) * Math.min(height[high], height[low]));
        	if(height[low] < height[high])low++;
        	else high--;
        }
        return max;
    }
 */

/**
 * 2018/1/9
 * 567. Permutation in String
 * 
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        int len1 = s1.length();
        int len2 = s2.length();
        if(len1 > len2)return false;
        int[] count = new int[26];
        for(int i = 0;i < len1;i++){
        	count[s1.charAt(i) - 'a']++;
        	count[s2.charAt(i) - 'a']--;
        }
        if(isAllZero(count))return true;
        for(int i = len1;i < len2;i++){
        	count[s2.charAt(i) - 'a']--;
        	count[s2.charAt(i - len1) - 'a']++;
        	if(isAllZero(count))return true;
        }
    	return false;
    }
    
    public boolean isAllZero(int[] count){
    	for(int i = 0;i < 26;i++){
    		if(count[i] != 0)return false;
    	}
    	return true;
    }
}
 */

/**
 * 2018/1/10
 * 73. Set Matrix Zeroes
 * 
    public void setZeroes(int[][] matrix) {
        if(matrix.length == 0 || matrix[0].length == 0)return;
        int row = matrix.length;
        int col = matrix[0].length;
        int[] rowArr = new int[row];
        int[] colArr = new int[col];
        for(int i = 0;i < row;i++){
        	for(int j = 0;j < col;j++){
        		if(matrix[i][j] == 0){
        			rowArr[i] = 1;
        			colArr[j] = 1;
        		}
        	}
        }
        for(int i = 0;i < row;i++){
        	if(rowArr[i] == 1){
        		for(int j = 0;j < col;j++){
        			matrix[i][j] = 0;
        		}
        	}
        }
        for(int j = 0;j < col;j++){
        	if(colArr[j] == 1){
        		for(int i = 0;i < row;i++){
        			matrix[i][j] = 0;
        		}
        	}
        }
    }
 */

/**
 * 2018/1/12
 * 80. Remove Duplicates from Sorted Array II
 * 
    public int removeDuplicates(int[] nums) {
        if(nums.length <= 2)return nums.length;
        int i = 2,j = 2;
        while(j < nums.length){
        	if(nums[j] != nums[i - 2])nums[i++] = nums[j++];
        	else j++;
        }
        return i;
    }
 */

/**
 * 2018/1/19
 * 762. Prime Number of Set Bits in Binary Representation
 * 
class Solution {
    public int countPrimeSetBits(int L, int R) {
        Set<Integer> set = primeSet();
        int res = 0;
        for(int i = L;i <= R;i++){
        	if(set.contains(countBits(i)))res++;
        }
        return res;
    }
    
    public Set<Integer> primeSet(){
    	Set<Integer> res = new HashSet<>();
    	boolean flag = true;
    	for(int i = 2;i < 50;i++){
    		flag = true;
    		for(int j = 2;j <= Math.sqrt(i);j++){
    			if(i % j == 0){
    				flag = false;
    				break;
    			}
    		}
    		if(flag)res.add(i);
    	}
    	return res;
    }
    
    public int countBits(int num){
    	int res = 0;
    	while(num > 0){
    		res += (num & 1);
    		num >>= 1;
    	}
    	return res;
    }
}

 * 2018/1/19
 * 763. Partition Labels
 * 
    public List<Integer> partitionLabels(String S) {
        int left = 0,right = 0;//区间左右
        List<Integer> res = new ArrayList<>();
        while(left < S.length()){
        	Set<Character> set = new HashSet<>();
        	for(int i = 0;i <= right;i++){
        		set.add(S.charAt(i));
        	}
        	boolean flag = true;
        	for(int j = S.length() - 1;j > right;j--){
        		if(set.contains(S.charAt(j))){
        			flag = false;
        			right = j;
        			break;
        		}
        	}
        	if(flag){
        		res.add(right - left + 1);
        		left = right + 1;
        		right = left;
        	}
        }
        return res;
    }
    
 * 2018/1/19
 * 331. Verify Preorder Serialization of a Binary Tree
 * 
class Solution {
    public boolean isValidSerialization(String preorder) {
        Stack<String> stack = new Stack<>();
        String[] order = preorder.split(",");
        int i = 0;
        while(i < order.length && !(stack.size() == 1 && stack.peek().equals("#"))){
        	stack.add(order[i++]);
        	while(check(stack));
        }
        if(i == order.length  && stack.size() == 1 && stack.peek().equals("#"))return true;
        return false;
    }
    
    public boolean check(Stack<String> stack){//将栈顶符合形式X##的进行更改
    	if(!stack.isEmpty() && stack.peek().equals("#")){
    		stack.pop();
    		if(!stack.isEmpty() && stack.peek().equals("#")){
    			stack.pop();
    			if(!stack.isEmpty() && !stack.peek().equals("#")){
    				stack.pop();
    				stack.add("#");
    				return true;
    			}
				stack.add("#");
    		}
			stack.add("#");
    	}
    	return false;
    }
}
 */

/**
 * 2018/1/20
 * 756. Pyramid Transition Matrix(Time Limit Exceeded)
 * 
class Solution {
    public boolean pyramidTransition(String bottom, List<String> allowed) {
        Map<String,Set<String>> allow = transverse(allowed);
        List<String> res = new ArrayList<>();
        res.add(bottom);
        for(int i = bottom.length();i >= 2;i--){
        	List<String> temp = generate(res, allow, i);
        	if(temp.size() == 0)return false;
        	res = temp;
        }
        return true;
    }
    
    public Map<String, Set<String>> transverse(List<String> allowed){//转换函数，将List转为HashMap便于查询
    	Map<String, Set<String>> res = new HashMap<>();
    	for(String allow : allowed){
    		if(res.containsKey(allow.substring(0, 2))){
    			res.get(allow.substring(0, 2)).add(allow.substring(2));
    		}
    		else{
    			Set<String> temp = new HashSet<>();
    			temp.add(allow.substring(2));
    			res.put(allow.substring(0, 2), temp);
    		}
    	}
    	return res;
    }
    
    public List<String> generate(List<String> primes, Map<String, Set<String>> allow,int n){
    	List<String> res = new ArrayList<>();
    	for(String prime : primes){
    		boolean flag = true;
    		for(int i = 0;i < n - 1;i++){
    			if(!allow.containsKey(prime.substring(i,i + 2))){
    				flag = false;
    				break;
    			}
    		}
    		if(!flag)continue;
    		List<String> tempRes = new ArrayList<>();
    		List<String> arrs = new ArrayList<>();
    		tempRes.addAll(allow.get(prime.substring(0, 2)));
    		for(int i = 1;i < n - 1;i++){
    			String sub = prime.substring(i, i + 2);
    			arrs = tempRes;
    			tempRes = new ArrayList<>();
    			for(String add : allow.get(sub)){
    				for(String arr : arrs){
    					tempRes.add(arr + add);
    				}
    			}
    		}
    		res.addAll(tempRes);
    	}
    	return res;
    }
}

 * 2018/1/20
 * 652. Find Duplicate Subtrees
 * 
class Solution {
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        List<TreeNode> res = new ArrayList<>();
        postOrder(root, res, new HashMap<>());
        return res;
    }
    
    public String postOrder(TreeNode root, List<TreeNode> res, HashMap<String, Integer> map){
    	if(root == null)return "#";
    	String series = root.val + "," + postOrder(root.left, res, map) + "," + postOrder(root.right, res, map);
    	if(map.getOrDefault(series, 0) == 1)res.add(root);
    	map.put(series, map.getOrDefault(series, 0) + 1);
    	return series;
    }
}

 * 2018/1/20
 * 752. Open the Lock
 * 
    public int openLock(String[] deadends, String target) {
        Set<String> visit = new HashSet<>();
        Set<String> dead = new HashSet<>(Arrays.asList(deadends));
        Queue<String> queue = new LinkedList<>();
        queue.add("0000");
        visit.add("0000");
        int level = 0;
        while(!queue.isEmpty()){
        	int size = queue.size();
        	while(size > 0){
        		String temp = queue.poll();
        		if(dead.contains(temp)){
        			size--;
        			continue;
        		}
        		if(temp.equals(target))return level;
        		StringBuffer s = new StringBuffer(temp);
        		for(int i = 0;i < 4;i++){
        			char c = s.charAt(i);
        			String s1 = s.substring(0, i) + (c == '9' ? 0 : c - '0' + 1) + s.substring(i + 1);
                    String s2 = s.substring(0, i) + (c == '0' ? 9 : c - '0' - 1) + s.substring(i + 1);
                    if(!visit.contains(s1) && !dead.contains(s1)) {
                        queue.offer(s1);
                        visit.add(s1);
                    }
                    if(!visit.contains(s2) && !dead.contains(s2)) {
                        queue.offer(s2);
                        visit.add(s2);
                    }
        		}
        		size--;
        	}
        	level++;
        }
        return -1;
    }
 */
package exercise;

import java.awt.Checkbox;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

class Solution_1231_To_0120 {
 
}

