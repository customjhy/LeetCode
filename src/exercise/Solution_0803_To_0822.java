/**
 * @author jhy
 * code from 8.3 to 8.22
 * 35 questions
 */


/**
 * 2017/8/3
 * 592. Fraction Addition and Subtraction
 * 
    public String fractionAddition(String expression) {
        List<Character> set = new ArrayList<Character>();
        if(expression.charAt(0) != '-'){
        	set.add('+');
        }
        for(int i = 0;i < expression.length();i++){
        	if(expression.charAt(i) == '+' || expression.charAt(i) == '-'){
        		set.add(expression.charAt(i));
        	}
        }
        int prenum = 0,preden = 1,i = 0;
        for(String expre : expression.split("(\\+)|(-)")){
        	if (expre.length() > 0) {
        		        	String[] fraction = expre.split("/");
        	int num = Integer.parseInt(fraction[0]);
        	int den = Integer.parseInt(fraction[1]);
        	int g = gcd(den,preden);
        	if(set.get(i++) == '+'){
        		prenum = (prenum * den + preden * num) / g;
        	}
        	else{
        		prenum = (prenum * den - preden * num) / g;
        	}
        	preden = preden * den / g;
        	g = Math.abs(gcd(prenum,preden));
        	prenum /= g;
        	preden /= g;
        	}
        }
        return prenum + "/" + preden;
    }
    
    public int gcd(int a,int b){
    	while(b != 0){
    		int t = b;
    		b = a % b;
    		a = t;
    	}
    	return a;
    }
    
    
 * 2017/8/3
 * 421. Maximum XOR of Two Numbers in an Array
 * 
    public int findMaximumXOR(int[] nums) {
    	int max = Integer.MIN_VALUE;
    	if(nums.length == 0)return 0;
    	if(nums.length == 1)return 0;
    	for(int i = 0;i < nums.length - 1;i++){
    		for(int j = i + 1;j < nums.length;j++){
    			if((nums[i] ^ nums[j]) > max){
    				max = nums[i] ^ nums[j];
    			}
    		}
    	}
    	return max;
    }
    
    
 * 2017/8/3
 * 481. Magical String
 * 
    public int magicalString(int n) {
        if(n < 1)return 0;
        if(n == 1)return 1;
        if(n == 2)return 1;
        if(n == 3)return 1;
        if(n == 4)return 2;
        int sum = 5;
        StringBuffer s = new StringBuffer("12211");
        int i = 3;
        while(sum < n){
        	if(s.charAt(i++) == '1'){
        		sum += 1;
        		if(s.charAt(s.length() - 1) == '1'){
        			s.append("2");
        		}
        		else{
        			s.append("1");
        		}
        	}
        	else{
        		sum += 2;
        		if(s.charAt(s.length() - 1) == '1'){
        			s.append("22");
        		}
        		else{
        			s.append("11");
        		}
        	}
        }
        return numOfMagical(s.toString(),n);
    }
    
    public int numOfMagical(String s,int n){
    	int sum = 0;
    	for(int i = 0;i < n;i++){
    		if(s.charAt(i) == '1')
    			sum++;
    	}
    	return sum;
    }
    
    
 * 2017/8/3
 * 216. Combination Sum III
 * 
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        combinationSum3(result,new ArrayList<Integer>(),k,1,n);
        return result;
    }
    
    public void combinationSum3(List<List<Integer>> result, List<Integer> temp,int k,int start, int n){
    	if(k == temp.size() && n == 0){
    		List<Integer> tempResult = new ArrayList<Integer>(temp);
    		result.add(tempResult);
    		return;
    	}
    	for(int i = start;i < 10;i++){
    		temp.add(i);
    		combinationSum3(result,temp,k,i + 1,n - i);
    		temp.remove(temp.size() - 1);
    	}
    }
    
    
 * 2017/8/3
 * 392. Is Subsequence
 * 
    public boolean isSubsequence(String s, String t) {
        int i = 0,j = 0;
        for(i = 0;i < s.length();i++){
        	boolean flag = true;
        	for(;j < t.length();j++){
        		if(t.charAt(j) == s.charAt(i)){
        			flag = false;
        			break;
        		}
        	}
        	if(flag){
        		return false;
        	}
        }
    	return true;
    }
 */

/**
 * 2017/8/8
 * 486. Predict the Winner
 * 
    public boolean PredictTheWinner(int[] nums) {
		return winner(nums, 0, nums.length - 1, 1) >= 0;
    }
    public int winner(int[] nums,int left,int right,int turn){
    	if(left == right)
    		return nums[left];
    	int a = turn * nums[left] + winner(nums,left + 1,right,-turn);
    	int b = turn * nums[right] + winner(nums,left,right - 1,-turn);
    	return turn * Math.max(turn * a, turn * b);
    }
    
    
 * 2017/8/8
 * 22. Generate Parentheses
 * 
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<String>();
        if(n <= 0)return result;
    	Set<String> set = parenthesis(n);
        result.addAll(set);
        return result;
    }
    public Set<String> parenthesis(int n){
    	if(n == 1){
    		Set<String> set = new HashSet<String>();
    		set.add("()");
    		return set;
    	}
    	Set<String> result = new HashSet<String>();
    	for(int i = 1;i < n;i++){
    		for(String str:parenthesis(i)){
    			for(String strstr : parenthesis(n - i)){
    				result.add(str + strstr);
    			}
    		}
    	}
    	for(String str:parenthesis(n - 1)){
    		result.add("(" + str + ")");
    	}
    	return result;
    }
    
    
 * 2017/8/8
 * 554. Brick Wall
 * 
    public int leastBricks(List<List<Integer>> wall) {
    	if(wall.size() == 0 || wall.get(0).size() == 0)return 0;
    	int count = 0;
    	Map<Integer,Integer> map = new HashMap<Integer,Integer>();
    	for(List<Integer> list : wall){
    		int length = 0;
    		for(int i = 0;i < list.size() - 1;i++){
    			length += list.get(i);
    			map.put(length, map.getOrDefault(length, 0) + 1);
    			count = Math.max(count, map.get(length));
    		}
    	}
    	return wall.size() - count;
    }
 */

/**
 * 2017/8/9
 * 378. Kth Smallest Element in a Sorted Matrix
 * 
    public int kthSmallest(int[][] matrix, int k) {
    	if(matrix.length == 0)return 0;
        int[] num = new int[matrix.length];
        int result = 0;
        for(int i = 0;i < k;i++){
        	result = smallest(matrix,num);
        }
        return result;
    }
    public int smallest(int[][] matrix,int[] num){
    	int minIndex = 0;
    	for(int i = 0;i < num.length;i++){
    		if(num[i] < num.length){
    			minIndex = i;
    			break;
    		}
    	}
    	for(int i = 0;i < matrix.length;i++){
    		if(num[i] < matrix.length && matrix[i][num[i]] < matrix[minIndex][num[minIndex]]){
    			minIndex = i;
    		}
    	}
    	return matrix[minIndex][num[minIndex]++];
    }
    
    
 * 2017/8/9
 * 423. Reconstruct Original Digits from English
 * 
	public String originalDigits(String s) {
	    int[] count = new int[10];
	    for (int i = 0; i < s.length(); i++){
	        char c = s.charAt(i);
	        if (c == 'z') count[0]++;
	        if (c == 'w') count[2]++;
	        if (c == 'x') count[6]++;
	        if (c == 's') count[7]++; //7-6
	        if (c == 'g') count[8]++;
	        if (c == 'u') count[4]++; 
	        if (c == 'f') count[5]++; //5-4
	        if (c == 'h') count[3]++; //3-8
	        if (c == 'i') count[9]++; //9-8-5-6
	        if (c == 'o') count[1]++; //1-0-2-4
	    }
	    count[7] -= count[6];
	    count[5] -= count[4];
	    count[3] -= count[8];
	    count[9] = count[9] - count[8] - count[5] - count[6];
	    count[1] = count[1] - count[0] - count[2] - count[4];
	    StringBuilder sb = new StringBuilder();
	    for (int i = 0; i <= 9; i++){
	        for (int j = 0; j < count[i]; j++){
	            sb.append(i);
	        }
	    }
	    return sb.toString();
	}
	
	
 * 2017/8/9
 * 12. Integer to Roman
 * 
    public String intToRoman(int num) {
        String M[] = {"","M","MM","MMM"};
        String C[] = {"","C","CC","CCC","CD","D","DC","DCC","DCCC","CM"};
        String X[] = {"","X","XX","XXX","XL","L","LX","LXX","LXXX","XC"};
        String I[] = {"","I","II","III","IV","V","VI","VII","VIII","IX"};
        return M[num / 1000] + C[(num % 1000) / 100] + X[(num % 100) / 10] + I[num % 10];
    }
    
    
 * 2017/8/9
 * 318. Maximum Product of Word Lengths
 * 
public class Solution {
    public int maxProduct(String[] words) {
    	int result = 0;
        Map<Integer,List<String>> map = new HashMap<Integer,List<String>>();
        List<Integer> length = new ArrayList<Integer>();
        for(int i = 0;i < words.length;i++){
        	if(map.containsKey(words[i].length())){
        		List<String> list = new ArrayList<String>(map.get(words[i].length()));
        		list.add(words[i]);
        		map.put(words[i].length(), list);
        	}
        	else{
        		List<String> list = new ArrayList<String>();
        		list.add(words[i]);
        		map.put(words[i].length(), list);
        		length.add(words[i].length());
        	}
        }
        Collections.sort(length, new Comparator<Integer>(){
        	public int compare(Integer a,Integer b){
        		return b - a;
        	}
        });
    	for(int i = 0;i < length.size();i++){
    		for(int j = 0;j <= i;j++){
    			for(String temp1 : map.get(length.get(i))){
    				for(String temp2 : map.get(length.get(j))){
    					if(notSimple(temp1,temp2)){
    						result = Math.max(result, length.get(i) * length.get(j));
    					}
    				}
    			}
    		}
    	}
    	return result;
    }
    public boolean notSimple(String temp1,String temp2){
    	Set<Character> set = new HashSet<Character>();
    	for(int i = 0;i < temp1.length();i++){
    		set.add(temp1.charAt(i));
    	}
    	for(int i = 0;i < temp2.length();i++){
    		if(set.contains(temp2.charAt(i))){
    			return false;
    		}
    	}
    	return true;
    }
}


 * 2017/8/9
 * 452. Minimum Number of Arrows to Burst Balloons
 * 
    public int findMinArrowShots(int[][] points) {
    	int result = 0;
    	Arrays.sort(points, new Comparator<int[]>(){
			public int compare(int[] o1, int[] o2) {
				if(o1[1] == o2[1])return o1[0] - o2[0];
				return o1[1] - o2[1];
			}
    	});
    	int i = 0;
    	while(i < points.length){
    		result++;
    		int temp = points[i][1];
    		while(i < points.length && temp >= points[i][0]){
    			i++;
    		}
    	}
    	return result;
    }
 */

/**
 * 2017/8/10
 * 230. Kth Smallest Element in a BST
 * 
    public int kthSmallest(TreeNode root, int k) {
        if(root == null || k <= 0)return -1;
    	PriorityQueue<Integer> queue = new PriorityQueue<Integer>((a,b) -> (b - a));
        Queue<TreeNode> que = new LinkedList<TreeNode>();
        que.add(root);
        while(!que.isEmpty()){
        	root = que.poll();
        	if(root.left != null)que.add(root.left);
        	if(root.right != null)que.add(root.right);
        	queue.add(root.val);
        	if(queue.size() > k)queue.poll();
        }
        if(queue.size() == k)return queue.peek();
        return -1;
    }
    
    
 * 2017/8/10
 * 494. Target Sum
 * 
public class Solution {
	int result = 0;
    public int findTargetSumWays(int[] nums, int S) {
        findSum(nums,0,S);
    	return result;
    }
    public void findSum(int[] nums,int k,int S){
    	if(k == nums.length){
    		if(S == 0) result++;
    		return;
    	}
    	findSum(nums,k + 1,S - nums[k]);
    	findSum(nums,k + 1,S + nums[k]);
    }
}


 */

/**
 * 2017/8/17
 * 46. Permutations
 * 
public class Solution {
	List<List<Integer>> result = new ArrayList<List<Integer>>();
	
    public List<List<Integer>> permute(int[] nums) {
        List<Integer> list = new ArrayList<Integer>();
        Set<Integer> set = new HashSet<Integer>();
        findArrange(nums,0,list,set);
        
        return result;
    }
    
    public void findArrange(int[] nums,int k,List<Integer> list,Set<Integer> set){
    	if(k == nums.length){
    		result.add(list);
    		return;
    	}
    	for(int i = 0;i < nums.length;i++){
    		if(!set.contains(nums[i])){
    			List<Integer> array = new ArrayList<Integer>(list);
    			array.add(nums[i]);
    			Set<Integer> tempSet = new HashSet<Integer>(set);
    			tempSet.add(nums[i]);
    			findArrange(nums, k + 1, array, tempSet);
    		}
    	}
    }
}


 * 2017/8/17
 * 131. Palindrome Partitioning
 * 
public class Solution {
    public List<List<String>> partition(String s) {
    	List<List<String>> result = new ArrayList<List<String>>();
    	isPartition(result,new ArrayList<String>(),s,0);
    	return result;
    }
    
    public void isPartition(List<List<String>> result, ArrayList<String> list,String s,int start){
    	if(start == s.length()){
    		result.add(new ArrayList<String>(list));
    		return;
    	}
    	for(int i = start;i < s.length();i++){
    		if(isPalindrome(s, start, i)){
    			list.add(s.substring(start,i + 1));
    			isPartition(result, list, s, i + 1);
    			list.remove(list.size() - 1);
    		}
    	}
    }
    
    public boolean isPalindrome(String s,int low,int high){
    	while(low < high){
    		if(s.charAt(low++) != s.charAt(high--)){
    			return false;
    		}
    	}
    	return true;
    }
}


 * 2017/8/17
 * 241. Different Ways to Add Parentheses
 * 
    public List<Integer> diffWaysToCompute(String input) {
        List<Integer> result = new ArrayList<Integer>();
        for(int i = 0;i < input.length();i++){
        	if(input.charAt(i) == '+' || input.charAt(i) == '-' || input.charAt(i) == '*'){
        		List<Integer> arr = diffWaysToCompute(input.substring(0,i));
        		List<Integer> brr = diffWaysToCompute(input.substring(i + 1));
        		for(int a : arr){
        			for(int b : brr){
        				if(input.charAt(i) == '+'){
        					result.add(a + b);
        				}
        				else if(input.charAt(i) == '-'){
        					result.add(a - b);
        				}
        				else{
        					result.add(a * b);
        				}
        			}
        		}
        	}
        }
        if(result.size() == 0){
        	result.add(Integer.valueOf(input));
        }
        return result;
    }
    
    
 * 2017/8/17
 * 328. Odd Even Linked List
 * 
    public ListNode oddEvenList(ListNode head) {
    	if(head == null || head.next == null)return head;
    	ListNode odd = head;
    	ListNode even = head.next;
    	ListNode temp = even;
    	while(even != null){
    		odd.next = even.next;
    		if(odd.next == null){
    			odd.next = temp;
    			return head;
    		}
    		odd = odd.next;
    		even.next = odd.next;
    		even = even.next;
    	}
        odd.next = temp;
        return head;

    }
 */


/**
 * 2017/8/18
 * 654. Maximum Binary Tree
 * 
class Solution {
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return construct(nums, 0, nums.length - 1);
    } 
    
    public TreeNode construct(int[] nums,int left,int right){
    	if(left > right)return null;
    	if(left == right)return new TreeNode(nums[left]);
    	int index = maxIndex(nums, left, right);
    	TreeNode root = new TreeNode(nums[index]);
    	root.left = construct(nums, left, index - 1);
    	root.right = construct(nums, index + 1, right);
    	return root;
    }
    
    public int maxIndex(int[] num,int left,int right){
    	int max = left;
    	for(int i = left + 1;i <= right;i++){
    		if(num[max] < num[i]){
    			max = i;
    		}
    	}
    	return max;
    }
}
 */

/**
 * 2017/8/19
 * 287. Find the Duplicate Number
 * 
    public int findDuplicate(int[] nums) {
    	if(nums.length == 0)return 0;
        for(int i = 1;i < nums.length;i++){
        	for(int j = 0;j < i;j++){
        		if((nums[j] ^ nums[i]) == 0){
        			return nums[j];
        		}
        	}
        }
        return nums[0];
    }
    
    
 * 2017/8/19
 * 337. House Robber III
 * 
    public int rob(TreeNode root) {
    	int[] result = arrayRob(root);
    	return Math.max(result[0], result[1]);
    }
    
    public int[] arrayRob(TreeNode root){
    	if(root == null)return new int[2];
    	int[] left = arrayRob(root.left);
    	int[] right = arrayRob(root.right);
    	int[] res = new int[2];
    	res[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
    	res[1] = root.val + left[0] + right[0];
    	return res;
    }
    
    
 * 2017/8/19
 * 650. 2 Keys Keyboard
 * 
    public int minSteps(int n) {
        int[] dp = new int[n + 1];
        for(int i = 2;i < dp.length;i++){
        	dp[i] = i;
        	for(int j = i - 1;j > 1;j--){
        		if(i % j == 0){
        			dp[i] = dp[j] + i / j;
        			break;
        		}
        	}
        }
        return dp[n];
    }
 */

/**
 * 2017/8/20
 * 583. Delete Operation for Two Strings
 * 
    public int minDistance(String word1, String word2) {
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        for(int i = 1;i < word1.length() + 1;i++){
        	for(int j = 1;j < word2.length() + 1;j++){
        		if(word1.charAt(i - 1) == word2.charAt(j - 1)){
        			dp[i][j] = 1 + dp[i - 1][j - 1];
        		}
        		else{
        			dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        		}
        	}
        }
        return word1.length() + word2.length() - 2 * dp[word1.length()][word2.length()];
    }
    
    
 * 2017/8/20
 * 398. Random Pick Index
 * 
class Solution {
	int[] array;
    public Solution(int[] nums) {
        array  = new int[nums.length];
        for(int i = 0;i < nums.length;i++){
        	array[i] = nums[i];
        }
    }
    
    public int pick(int target) {
    	Random rand =new Random(25);
        int num = 0;
        for(int i = 0;i < array.length;i++){
        	if(array[i] == target)num++;
        }
        int j = 0;
        int result = -1;
        for(int i = 0;i < num;i++){
        	while(array[j] != target){
        		j++;
        	}
        	if(rand.nextInt(i + 1) == i){
        		result = j;
        	}
        	j++;
        }
        return result;
    }
}
 */

/**
 * 2017/8/21
 * 516. Longest Palindromic Subsequence
 * 
    public int longestPalindromeSubseq(String s) {
    	if(s.length() == 0)return 0;
    	int[][] dp = new int[s.length()][s.length()];
    	for(int i = 0;i < s.length();i++){
    		dp[i][i] = 1;
    	}
    	for(int i = s.length() - 2;i >= 0;i--){
    		for(int j = i + 1;j < s.length();j++){
    			if(s.charAt(i) == s.charAt(j)){
    				dp[i][j] = 2 + dp[i + 1][j - 1];
    			}
    			else{
    				dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
    			}
    		}
    	}
    	return dp[0][s.length() - 1];
    }
    
    
 * 2017/8/21
 * 524. Longest Word in Dictionary through Deleting
 * 
class Solution {
	public String findLongestWord(String s, List<String> d) {
		String result = new String("");
		int length = 0;
		for (String str : d) {
			if (isWord(str, s)) {
				if (str.length() > length) {
					result = str;
					length = str.length();
				} else if (str.length() == length) {
					if (str.compareTo(result) < 0) {
						result = str;
					}
				}
			}
		}
		return result;
	}
    
    public boolean isWord(String str,String s) {
    	if(str.length() == 0 ||s.length() == 0)return false;
		int i = 0,j = 0;
		while(i != str.length()){
			while(j < s.length() && str.charAt(i) != s.charAt(j)){
				j++;
			}
			if(j == s.length())return false;
			i++;
			j++;
		}
    	return true;
	}
}


 * 2017/8/21
 * 319. Bulb Switcher
 * 
    public int bulbSwitch(int n) {
        return (int)Math.sqrt(n);
    }
    
    
 * 2017/8/21
 * 657. Judge Route Circle
 * 
    public boolean judgeCircle(String moves) {
        int horizon = 0;
        int vertical = 0;
        for(int i = 0;i < moves.length();i++){
        	if(moves.charAt(i) == 'L')horizon--;
        	else if(moves.charAt(i) == 'R')horizon++;
        	else if(moves.charAt(i) == 'U')vertical++;
        	else vertical--;
        }
        return horizon == 0 && vertical == 0;
    }
    
    
 * 2017/8/21
 * 637. Average of Levels in Binary Tree
 * 
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> result = new ArrayList<Double>();
        if(root == null)return result;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        while(!queue.isEmpty()){
        	int size = queue.size();
        	double sum = 0;
        	for(int i = 0;i < size;i++){
        		root = queue.poll();
        		sum += root.val;
        		if(root.left != null)queue.add(root.left);
        		if(root.right != null)queue.add(root.right);
        	}
        	result.add((sum / size));
        }
        return result;
    }
 */

/**
 * 2017/8/22
 * 653. Two Sum IV - Input is a BST
 * 
    public boolean findTarget(TreeNode root, int k) {
        List<Integer> list = new ArrayList<Integer>();
        findList(root,list);
        int index = 0;
        for(int i = 0;i < list.size();i++){
        	if(list.get(i) > (k + 1) / 2 -1){
        		index = i;
        		break;
        	}
        }
        for(int i = 0;i < index;i++){
        	if(list.contains(k - list.get(i))){
        		return true;
        	}
        }
        return false;
    }
    
    public void findList(TreeNode root,List<Integer> list){
    	if(root == null)return;
    	findList(root.left, list);
    	list.add(root.val);
    	findList(root.right, list);
    }
    
    
 * 2017/8/22
 * 538. Convert BST to Greater Tree
 * 
class Solution {
	int i = 0;
    public TreeNode convertBST(TreeNode root) {
    	List<Integer> list = new ArrayList<Integer>();
        findList(root,list);
        for(int i = 0;i < list.size();i++){
        	int sum = list.get(i);
        	for(int j = i + 1;j < list.size();j++){
        		sum += list.get(j);
        	}
        	list.set(i, sum);
        }
        convert(root,list);
        return root;
    }
    
    public void findList(TreeNode root,List<Integer> list){
    	if(root == null)return;
    	findList(root.left, list);
    	list.add(root.val);
    	findList(root.right, list);
    }
    
    public void convert(TreeNode root,List<Integer> list){
    	if(root == null)return;
    	convert(root.left, list);
    	root.val = list.get(i++);
    	convert(root.right, list);
    }
}


 * 2017/8/22
 * 645. Set Mismatch
 * 
    public int[] findErrorNums(int[] nums) {
    	int index = 0;
        for(int i = 0;i < nums.length;i++){
        	index = Math.abs(nums[i]) - 1;
        	nums[index] = -nums[index];
        }
        int[] result = new int[2];
        int j = 0;
        for(int i = 0;i < nums.length;i++){
        	if(nums[i] > 0){
        		result[j++] = i + 1;
        	}
        }
        boolean flag = true;
        for(int i = 0;i < nums.length;i++){
        	if(result[0] == Math.abs(nums[i])){
        		flag = false;
        		break;
        	}
        }
        if(flag){
        	int temp = result[0];
        	result[0] = result[1];
        	result[1] = temp;
        }
        return result;
    }
    
    
 * 2017/8/22
 * 643. Maximum Average Subarray I
 * 
    public double findMaxAverage(int[] nums, int k) {
    	if(nums.length < k)return 0;
        double max = 0;
        double temp = 0;
        for(int i = 0;i < k;i++){
        	max += nums[i];
        	temp = max;
        }
        for(int i = k;i < nums.length;i++){
        	temp = temp + nums[i] - nums[i - k];
        	max = Math.max(max, temp);
        }
        return max / k;
    }
    
    
 * 2017/8/22
 * 655. Print Binary Tree
 * 
class Solution {
    public List<List<String>> printTree(TreeNode root) {
        int H = height(root);
        int row = H;
        int col = (int)Math.pow(2, H) - 1;
        List<List<String>> result = new ArrayList<List<String>>();
        List<String> temp = new ArrayList<String>();
        for(int i = 0;i < col;i++){
        	temp.add("");
        }
        for(int i = 0;i < row;i++){
        	result.add(new ArrayList<String>(temp));
        }
        print(result,root,0,row,0,col - 1);
        return result;
    }
    
    public void print(List<List<String>> result,TreeNode root,int a,int row,int i,int j){
    	if(a == row || root == null)return;
    	result.get(a).set((i + j) / 2, Integer.toString(root.val));
    	print(result, root.left, a + 1, row, i, (i + j) / 2 - 1);
    	print(result, root.right, a + 1, row, (i + j) / 2 + 1, j);
    }
    
    public int height(TreeNode root){
    	if(root == null)return 0;
    	return 1 + Math.max(height(root.left), height(root.right));
    }
}
 */

package exercise;

import java.util.ArrayList;
import java.util.List;


/*    public int[][] imageSmoother(int[][] M) {
if(M.length == 0 || M.length == 1 && M[0].length == 1) return M;
int row = M.length;
int col = M[0].length;
int[][] result = new int[row][col];
for(int i = 1;i < row - 1;i++){
	for(int j = 1;j < col - 1;j++){
		int sum = 0;
		for(int k = i - 1;k <= i + 1;k++){
			for(int l = j - 1;l <= j + 1;l++){
				sum += M[k][l];
			}
		}
		result[i][j] = sum / 9;
	}
}


return result;
}*/












