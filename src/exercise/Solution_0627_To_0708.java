/**
 * @author jhy
 * code from 6.27 to 7.8
 * 40 questions
 */


/**
 * 2017/6/27
 * 561. Array Partition I
 * 
	public int arrayPairSum(int[] nums) {
        Arrays.sort(nums);
        int sum = 0;
		for(int i = 0;i < nums.length;i += 2){
			sum += nums[i];
		}
		return sum;
    }
*/

/**
 * 2017/6/28
 * 566. Reshape the Matrix
 * 
public int[][] matrixReshape(int[][] nums, int r, int c) {
	if(nums.length==0){
		return nums;
	}
	else if(nums.length * nums[0].length != r * c){
		return nums;
	}
	else{
		int[][] result = new int[r][c];
		int length = nums[0].length;
		int column = 0;
		int row = 0;
		for(int i = 0;i < r;i++){
			for(int j = 0;j < c;j++){
				result[i][j] = nums[row][column++];
				if(column == length){
					column = 0;
					row++;
				}
			}
		}
		return result;
	}
}


 * 2017/6/28
 * 575. Distribute Candies
 *	
	public int distributeCandies(int[] candies) {
        Arrays.sort(candies);
        int temp = candies[0];
        int num = 1;
        for(int i = 1;i < candies.length;i++){
        	if(temp != candies[i]){
        		temp = candies[i];
        		num++;
        	}
        }
        if(num > candies.length / 2)
        	return candies.length / 2;
        else 
        	return num;
    }
    
    
 * 2017/6/28
 * 606. Construct String from Binary Tree
 *    
	public void recursive(StringBuilder result,TreeNode t){
		if(t != null){
			result.append(t.val);
			if(t.left != null || t.right != null){
				result.append("(");
				recursive(result,t.left);
				result.append(")");
				if(t.right != null){
					result.append("(");
					recursive(result,t.right);
					result.append(")");
				}
			}
		}
	}
	
	public String tree2str(TreeNode t) {
		StringBuilder result = new StringBuilder();
		recursive(result, t);
		return result.toString();
    }
	
     
 * 2017/6/28
 * 598. Range Addition II
 *    
	public int maxCount(int m, int n, int[][] ops) {
		int row = m;
		int colunm = n;
		for(int[] pair:ops){
			row = Math.min(row, pair[0]);
			colunm = Math.min(colunm, pair[1]);
		}
		return row * colunm;
    }
 */


/**
 * 2017/6/29
 * 599. Minimum Index Sum of Two Lists
 * 
    public String[] findRestaurant(String[] list1, String[] list2) {
        ArrayList<String> result = new ArrayList<String>();
        int min = Integer.MAX_VALUE;
        for(int i = 0;i < list1.length;i++){
        	for(int j = 0;j < list2.length;j++){
        		if(list1[i].equals(list2[j])){
        			if(min > i + j){
        				result.clear();
        				min = i + j;
        				result.add(list1[i]);
        			}
        			else if(min == i + j){
        				result.add(list1[i]);
        			}
        		}
        	}
        }
        String[] res = new String[result.size()];
        for(int i = 0;i < result.size();i++){
        	res[i] = result.get(i);
        }
        return res;
    }
    
    
 * 2017/6/29
 * 628. Maximum Product of Three Numbers
 *    
    public int maximumProduct(int[] nums) {
        int[] result = new int[5];
        int temp = 0;
        for(int i = 0;i < 5;i++){
        	result[i] = 0;
        }
        int flag = 1;
        for(int i = 0;i < nums.length;i++){
        	if(nums[i] > result[2]){
        		result[2] = nums[i];
        		sortUp(result);
        	}
        	if(nums[i] < result[4]){
        		result[4] = nums[i];
        		sortDown(result);
        	}
        	if(nums[i] > 0){
        		flag = 0;
        	}
        }
        if(flag == 1){
        	for(int i = 0;i < 3;i++){
        		result[i] = Integer.MIN_VALUE;
        	}
        	for(int i = 0;i < nums.length;i++){
        		if(nums[i] > result[2]){
        			result[2] = nums[i];
        			sortUp(result);
        		}
        	}
        	return result[0] * result[1] * result[2];
        }
        int first = result[0] * result[1] * result[2];
        int second = result[0] * result[3] * result[4];
        return (first > second)? first : second;
    }
    
    public void sortUp(int[] num){
    	int temp = num[0];
    	if(num[1] > num[0]){
    		temp = num[1];
    		num[1] = num[0];
    		num[0] = temp;
    	}
    	if(num[2] > num[0]){
    		temp = num[2];
    		num[2] = num[0];
    		num[0] = temp;
    	}
    	if(num[2] > num[1]){
    		temp = num[2];
    		num[2] = num[1];
    		num[1] = temp;
    	}
    }
    
    public void sortDown(int[] num){
    	int temp = 0;
    	if(num[4] < num[3]){
    		temp = num[4];
    		num[4] = num[3];
    		num[3] = temp;
    	}
    }
*/


/**
 * 2017/6/30
 * 447. Number of Boomerangs
 * 
	public int numberOfBoomerangs(int[][] points) {
    	int length = points.length;
    	if(length <= 2){
    		return 0;
    	}
        double[][] result = new double[length][length];
        for(int i = 0;i < length;i++){
        	for(int j = 0;j <= i;j++){
        		result[i][j] = this.distance(points[i],points[j]);
        	}
        }
        for(int i = 0;i < length;i++){
        	for(int j = 0;j < i;j++){
        		result[j][i] = result[i][j];
        	}
        }
        
        for(int i = 0;i < length;i++){
        	Arrays.sort(result[i]);
        }
        
    	int sum = 0;
    	int equalNum = 1;
    	double temp;
    	for(int i = 0;i < length;i++){
    		temp = result[i][1];
    		equalNum = 1;
    		for(int j = 2;j < length;j++){
    			if(equal(temp, result[i][j])){
    				equalNum++;
    			}
    			else{
    				temp = result[i][j];
    				sum = sum + equalNum * (equalNum - 1);
    				equalNum = 1;
    			}
    		}
    		if(equalNum > 1){
    			sum = sum + equalNum * (equalNum - 1);
    		}
    	}
    	return sum;
    }
    
    public double distance(int[] arr,int[] brr){
    	return Math.sqrt((arr[0] - brr[0]) * (arr[0] - brr[0]) + (arr[1] - brr[1]) * (arr[1] - brr[1]));
    }
    
    public boolean equal(double num1,double num2)
    {
    	if((num1 - num2 > -0.0000001) && (num1 - num2)<0.00000001) 
    		return true;
    	else 
    		return false;
    }
    
    
 * 2017/6/30
 * 594. Longest Harmonious Subsequence
 *	
    public int findLHS(int[] nums) {
    	if(nums.length == 0){
    		return 0;
    	}
    	if(nums.length == 1){
    		return 0;
    	}
        Arrays.sort(nums);
        int max = 0;
        int temp = nums[0];
        int tempMax = 1;
        int record = 0;
        boolean flag = true;//判断是否为加一后的
        boolean flag2 = false;//判断是否有加一
        for(int i = 1;i < nums.length;i++){
        	if(temp == nums[i]){
        		tempMax++;
        	}
        	else if(Math.abs(temp - nums[i]) == 1 && flag){
        		temp = nums[i];
        		record = tempMax;
        		tempMax++;
        		flag = false;
        		flag2 = true;
        	}
        	else if(Math.abs(temp - nums[i]) == 1 && !flag){
        		temp = nums[i];
        		max = (tempMax > max)? tempMax : max;
        		tempMax = tempMax - record + 1;
        		record = tempMax - 1;
        	}
        	else if(!flag){
        		max = (tempMax > max)? tempMax : max;
        		temp = nums[i];
        		flag = true;
        		tempMax = 1;
        	}
        	else{
        		temp = nums[i];
        		flag = true;
        		tempMax = 1;
        	}
        }
        if(!flag)
        	max = (tempMax > max)? tempMax : max;
        if(flag2)
        	return max;
        else 
        	return 0;
    }
    
    
 * 2017/6/30
 * 191. Number of 1 Bits
 *	
    public int hammingWeight(int n) {
        int result = 0;
        for(;n != 0;n = n & (n-1)){
        	result++;
        }
    	return result;
    }
*/


/**
 * 2017/7/1
 * 263. Ugly Number
 * 
	public boolean isUgly(int num) {
        if(num <= 0){
        	return false;
        }
        while(num % 2 == 0){
        	num = num / 2;
        }
        while(num % 3 == 0){
        	num = num /3;
        }
        while(num % 5 == 0){
        	num = num /5;
        }
        if(num == 1){
        	return true;
        }
        return false;
    }
    
    
 * 2017/7/1
 * 21. Merge Two Sorted Lists
 * 
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode result = new ListNode(0);
        ListNode temp = result;
        while(l1 != null && l2 != null){
        	if(l1.val < l2.val){
        		temp.next = l1;
        		l1 = l1.next;
        		temp = temp.next;
        	}
        	else{
        		temp.next = l2;
        		l2 = l2.next;
        		temp = temp.next;
        	}
        }
    	if(l1 != null){
    		temp.next = l1;
    	}
    	if(l2 != null){
    		temp.next = l2;
    	}
    	return result.next;
    }
	
	
 * 2017/7/1
 * 27. Remove Element
 * 
    public int removeElement(int[] nums, int val) {
        int result = nums.length;
        for(int i = 0;i < result;i++){
        	if(nums[i] == val){
        		nums[i] = nums[--result];
        		i--;
        	}
        }
        return result;
    }
    
    
 * 2017/7/1
 * 235. Lowest Common Ancestor of a Binary Search Tree
 * 
	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(this.isSearch(p, q)){
        	return p;
        }
        if(this.isSearch(q, p)){
        	return q;
        }
		TreeNode temp = null;
		TreeNode result = null;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
		int length = 0;
		queue.add(root);
		while(!queue.isEmpty()){
			length = queue.size();
			for(int i = 0;i < length;i++){
				temp = queue.poll();
				if(this.isSearch(temp, p) && this.isSearch(temp, q)){
					result = temp;
				}
				if(temp == p || temp == q){
					return result;
				}
				if(temp.left != null){
					queue.add(temp.left);
				}
				if(temp.right != null){
					queue.add(temp.right);
				}
			}
		}
        return result;
    }
	
	public boolean isSearch(TreeNode root,TreeNode aim){
		while(root != null){
			if(root.val > aim.val){
				root = root.left;
			}
			else if(root.val < aim.val){
				root = root.right;
			}
			else{
				return true;
			}
		}
		return false;
	}
*/

/**
 * 2017/7/2
 * 101. Symmetric Tree
 * 
    public boolean isSymmetric(TreeNode root) {
    	if(root == null)return true;
    	Queue<TreeNode> result = new LinkedList<TreeNode>();
    	ArrayList<Integer> arr = new ArrayList<Integer>();
    	result.add(root);
    	TreeNode temp = null;
    	int length;
    	while(!result.isEmpty()){
    		length = result.size();
    		arr.clear();
    		for(int i = 0;i < length;i++){
    			temp = result.poll();
    			if(temp.left == null){
    				arr.add(0);
    			}
    			else{
    				arr.add(temp.left.val);
    				result.add(temp.left);
    			}
    			if(temp.right == null){
    				arr.add(0);
    			}
    			else{
    				arr.add(temp.right.val);
    				result.add(temp.right);
    			}
    		}
    		length = arr.size();
    		for(int i = 0;i < length / 2;i++){
    			if(arr.get(i) != arr.get(length - 1 - i))
    				return false;
    		}
    	}
    	return true;
    }
*/


/**
 * 2017/7/4
 * 108. Convert Sorted Array to Binary Search Tree
 * 
	public TreeNode sortedArrayToBST(int[] nums) {
        TreeNode result = sortedArray(nums, 0, nums.length - 1);
        return result;
    }
	
	public TreeNode sortedArray(int[] nums,int left,int right){
		if(left > right){
			return null;
		}
		int mid = (left + right) / 2;
		TreeNode temp = new TreeNode(nums[mid]);
		temp.left = sortedArray(nums, left, mid - 1);
		temp.right = sortedArray(nums, mid + 1, right);
		return temp;
	}
	
	
 * 2017/7/4
 * 345. Reverse Vowels of a String
 * 
    public String reverseVowels(String s) {
        char[] temp = s.toCharArray();
        List<Integer> tempArr = new ArrayList<Integer>();
        int index = -1;
        do{
        	index = indexOfVowel(temp, index);
        	if(index != -1){
        		tempArr.add(index);
        	}
        }while(index != -1);
        //Sort comparator = new Sort();
        Collections.sort(tempArr);
        char ch;
        int length = tempArr.size();
        for(int i = 0;i < length / 2;i++){
        	ch = temp[tempArr.get(i)];
        	temp[tempArr.get(i)] = temp[tempArr.get(length - 1 - i)];
        	temp[tempArr.get(length - i - 1)] = ch;
        }
        return String.valueOf(temp);
    }
    
    public int indexOfVowel(char[] temp,int flag){
    	char[] compare = new char[]{'a','e','i','o','u','A','E','I','O','U'};
    	for(int i = flag + 1;i < temp.length;i++){
    		for(int j = 0;j < 10;j++){
    			if(temp[i] == compare[j]){
    				return i;
    			}
    		}
    	}
    	return -1;
    }
*/
    
/**
 * 2017/7/5
 * 342. Power of Four
 * 
    public boolean isPowerOfFour(int num) {
        return (num > 0) &&((Math.log10(num) / Math.log10(4)) % 1 ==0);
    }
    
    
 * 2017/7/5
 * 66. Plus One
 * 
    public int[] plusOne(int[] digits) {
    	int flag0 = 1; //所有数字均为零
    	int flag9 = 1; // 所有数字均为9
        for(int i = 0;i < digits.length;i++){
        	if(digits[i] != 0){
        		flag0 = 0;
        		break;
        	}
        }
        if(flag0 == 1){
        	int[] arr = new int[1];
        	arr[0] = 1;
        	return arr;
        }
        for(int i = 0;i < digits.length;i++){
        	if(digits[i] != 9){
        		flag9 = 0;
        		break;
        	}
        }
        if(flag9 == 1){
        	int[] arr = new int[digits.length + 1];
        	arr[0] = 1;
        	for(int i = 1;i < digits.length + 1;i++){
        		arr[i] = 0;
        	}
        	return arr;
        }
        for(int i = digits.length - 1;i >= 0;i--){
        	digits[i]++;
        	if(digits[i] >= 10){
        		digits[i] = 0;
        	}
        	else{
        		return digits;
        	}
        }
        return digits;
    }
    
    
 * 2017/7/5
 * 459. Repeated Substring Pattern
 * 
    public boolean repeatedSubstringPattern(String s) {
        if(s.length() == 1)return false;
    	char[] ch = s.toCharArray();
        int flag = 1;//全部为相同字母
        for(int i = 1;i < ch.length;i++){
        	if(ch[i] != ch[0]){
        		flag = 0;
        		break;
        	}
        }
        if(flag == 1){
        	return true;
        }
        flag = 1;
        for(int size = 2;size <= s.length() / 2;size++){
        	if(s.length() % size != 0)continue;
        	flag = 1;
        	for(int i = size;i < ch.length;i++){
            	if(ch[i % size] != ch[i]){
            		flag = 0;
            		break;
            	}
            }
        	if(flag == 1)
        		return true;
        }
        return false;
    }
    
    
 * 2017/7/5
 * 367. Valid Perfect Square
 * 
    public boolean isPerfectSquare(int num) {
        if(num == 1 || num == 4)return true;
        for(int i = 3;i < num / 2;i++){
        	if(i * i > num){
        		break;
        	}
        	if(i * i == num){
        		return true;
        	}
        }
        return false;
    }
    
    
 * 2017/7/5
 * 118. Pascal's Triangle
 * 
    public List<List<Integer>> generate(int numRows) {
    	List<List<Integer>> result = new ArrayList<List<Integer>>();
    	if(numRows <= 0)return result;
        List<Integer> temp = new ArrayList<Integer>();
        temp.add(1);
        result.add(temp);
        if(numRows == 1)return result;
        List<Integer> pre = temp;
        for(int j = 1; j < numRows;j++){
        	temp = new ArrayList<Integer>();
        	temp.add(1);
        	for(int i = 0;i < pre.size() - 1;i++){
        		temp.add(pre.get(i) + pre.get(i + 1));
        	}
        	temp.add(1);
        	result.add(temp);
        	pre = temp;
        }
        return result;
    }
    
    
 * 2017/7/5
 * 198. House Robber
 * 
    public int rob(int[] nums) {
    	if(nums.length == 0)return 0;
        int rob = nums[0];
        int notRob = 0;
        int cur = 0;
        for(int i = 1;i < nums.length;i++){
        	cur = notRob;
        	notRob = rob;
        	rob = Math.max(rob, cur + nums[i]);
        }
        return Math.max(rob, notRob);
    }
*/


/**
 * 2017/7/6
 * 119. Pascal's Triangle II
 * 
    public List<Integer> getRow(int rowIndex) {
        List<Integer> result = new ArrayList<Integer>();
        if(rowIndex < 0)return result;
        for(int i = 0;i <= rowIndex;i++){
        	result.add(PLZH(rowIndex, i));
        }
        return result;
    }

    //进行排列组合
    public int PLZH(int n,int m){
    	if(n < m)return -1;
    	double result = 1;
    	for(int i = n - m + 1;i <= n;i++){
    		result *= i;
    	}
    	for(int i = 2;i <= m;i++){
    		result /= i;
    	}
    	return (int)(result + 0.1);
    }
    
    
 * 2017/7/6
 * 119. Pascal's Triangle II
 * 
public class Solution {
	private int curVal = -1;//当前值
	private int maxNum = 0;//出现次数最大
	private int curNum = 0;//当前出现次数
	private int length = 0;//出现次数最大的数据个数
	
	private int result[];//保存结果数组
	
    public int[] findMode(TreeNode root) {
        inOrder(root);
        result = new int[length];
        if(root == null)
    		return result;
        curVal = root.val;
        curNum = 0;
        inOrder(root);
        return result;
    }
    
    public void maxMode(int temp){
    	if(temp != curVal){
    		curVal = temp;
    		curNum = 0;
    	}
    	curNum++;
    	if(curNum > maxNum){
    		maxNum = curNum;
    		length = 1;
    	}
    	else if(curNum == maxNum){
    		if(result == null){
    			length++;
    		}
    		else{
    			result[--length] = temp;
    		}
    	}
    }
    
    public void inOrder(TreeNode root){
    	if(root == null)return;
    	inOrder(root.left);
    	maxMode(root.val);
    	inOrder(root.right);
    }  
}


 * 2017/7/6
 * 617. Merge Two Binary Trees
 * 
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if(t1 != null && t2 != null){
        	if(t1 == t2){
        		return t1;
        	}
        	else{
        		t1.val += t2.val; 
            	t1.left = mergeTrees(t1.left, t2.left);
            	t1.right = mergeTrees(t1.right, t2.right);
        	}
        }
        else if(t1 != null && t2 == null){
        	return t1;
        }
        else if(t1 == null && t2 != null){
        	t1 = t2;
        }
        return t1;
    }
    
    
 * 2017/7/6
 * 110. Balanced Binary Tree
 * 
    public boolean isBalanced(TreeNode root) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        while(!stack.isEmpty() || root != null){
        	while(root != null){
        		stack.push(root);
            	if(Math.abs(heightOfTree(root.left) - heightOfTree(root.right)) > 1){
            		return false;
            	}
            	root = root.left;
        	}
        	root = stack.pop();
        	root = root.right;
        }
        return true;
    }
    
    public int heightOfTree(TreeNode root){
    	if(root == null)return 0;
    	return Math.max(heightOfTree(root.left), heightOfTree(root.right)) + 1;
    }
    
    
 * 2017/7/6
 * 437. Path Sum III
 * 
    public int pathSum(TreeNode root, int sum) {
    	if(root == null)return 0;
        return pathSumRoot(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    }
    
    public int pathSumRoot(TreeNode root,int sum){
    	if(root == null)return 0;
    	return (root.val == sum ? 1 : 0) + pathSumRoot(root.left, sum - root.val) + pathSumRoot(root.right, sum - root.val);
    }
*/


/**
 * 2017/7/7
 * 434. Number of Segments in a String
 * 
    public int countSegments(String s) {
    	int sum = 0;
    	int i = 0;
    	while(i < s.length() && s.charAt(i) != ' '){
    		i++;
    	}
    	if(i != 0)sum++;
    	for(;i < s.length();i++){
    		if(s.charAt(i) != ' ' && s.charAt(i - 1) == ' '){
    			sum++;
    		}
    	}
    	return sum;
    }
    
    
 * 2017/7/7
 * 257. Binary Tree Paths
 * 
public class Solution {
    public List<String> result = new ArrayList<String>();
    public List<String> binaryTreePaths(TreeNode root) {
    	if(root == null)return result;
    	binaryPath(root,"" + root.val);
    	return result;
    }
    
    public void binaryPath(TreeNode root, String str){
    	if(root.left == null && root.right == null)result.add(str);
    	else if(root.left != null && root.right == null){
    		binaryPath(root.left, str + "->" + root.left.val);
    	}
    	else if(root.left == null && root.right != null){
    		binaryPath(root.right, str + "->" + root.right.val);
    	}
    	else{
    		binaryPath(root.left, str + "->" + root.left.val);
    		binaryPath(root.right, str + "->" + root.right.val);
    	}
    }
}


 * 2017/7/7
 * 441. Arranging Coins
 * 
    public int arrangeCoins(int n) {
        int i = 1;
        while(n >= i){
        	n = n - i;
        	i++;
        }
        return i - 1;
    }
    

 * 2017/7/7
 * 172. Factorial Trailing Zeroes
 * 
    public int trailingZeroes(int n) {
        int sum = 0;
        while(n > 0){
        	n = n / 5;
        	sum += n;
        }
        return sum;
    }
    
    
 * 2017/7/7
 * 141. Linked List Cycle
 * 
    public boolean hasCycle(ListNode head) {
    	if(head == null)return false;
        ListNode first = head;
        ListNode second = head;
        while(first.next != null && second.next != null && second.next.next != null){
        	if(first.next == second.next.next)return true;
        	first = first.next;
        	second = second.next.next;
        }
        return false;
    }
*/    
    
/**
 * 2017/7/8
 * 26. Remove Duplicates from Sorted Array
 * 
    public int removeDuplicates(int[] nums) {
    	if(nums.length == 0)return 0;
        int j = 1;
        int i = 0;
        for(i = 0;i + j < nums.length;i++){
        	while(nums[i] == nums[i + j]){
        		j++;
        		if(i + j == nums.length)
        			return i + 1;
        	}
        	nums[i + 1] = nums[i + j];
        }
        return i + 1;
    }
    
    
 * 2017/7/8
 * 9. Palindrome Number
 * 
    public boolean isPalindrome(int x) {
        if(x < 0 || (x !=0 && x % 10 == 0))return false;
        int temp = 0;
        while(temp < x){
        	temp = temp * 10 + x % 10;
        	x = x / 10;
        }
        if(x != temp / 10 && x != temp)return false;
        return true;
    }
    
    
 * 2017/7/8
 * 38. Count and Say
 * 
    public String countAndSay(int n) {
    	String result = "1";
        if(n <= 0){
        	return "";
        }
    	for(int i = 1;i < n;i++){
    		result = countSay(result);
    	}
    	return result;
    }
    
    public String countSay(String str){
    	String result = new String("");
    	int i = 0;
    	int j = 0;
    	while(i < str.length()){
    		j = 0;
    		while(i + j < str.length() && str.charAt(i) == str.charAt(i + j)){
    			j++;
    		}
    		result = result + j + str.charAt(i);
    		if(i + j == str.length())
    			return result;
    		i = i + j;
    	}
    	return result;
    }
    
    
 * 2017/7/8
 * 1. Two Sum
 * 
    public int[] twoSum(int[] nums, int target) {
    	if(nums.length < 2)return null;
    	int[] result = new int[2];
        Map<Integer,Integer> store = new HashMap<Integer,Integer>();
        for(int i = 0;i < nums.length;i++){
        	store.put(nums[i], i);
        }
        for(int i = 0;i < nums.length;i++){
        	if(store.get(target - nums[i]) != null && store.get(target - nums[i]) != i){
        		result[0] = i;
        		result[1] = store.get(target - nums[i]);
        		return result;
        	}
        }
        return null;
    }
    
    
 * 2017/7/8
 * 112. Path Sum
 * 
    public boolean hasPathSum(TreeNode root, int sum) {
    	if(root == null)return false;
        if(root.left == null && root.right == null){
        	if(root.val == sum)return true;
        	else return false;
        }
        else if(root.left == null && root.right != null){
        	return hasPathSum(root.right, sum - root.val);
        }
        else if(root.left != null && root.right == null){
        	return hasPathSum(root.left, sum - root.val);
        }
        else{
        	return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
        }
    }
*/




package exercise;

import java.util.HashMap;
import java.util.Map;

import javax.print.attribute.HashPrintRequestAttributeSet;

public class Solution_0627_To_0708 {
	
}
