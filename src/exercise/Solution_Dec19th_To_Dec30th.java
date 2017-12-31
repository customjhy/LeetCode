/**
 * @author jhy
 * code from 12.19 to 12.30
 * 36 questions
 */

/**
 * 2017/12/19
 * 611. Valid Triangle Number
 * 
    public int triangleNumber(int[] nums) {
        int res = 0;
        Arrays.sort(nums);
        for(int i = 0;i < nums.length - 2;i++){
        	int k = i + 2;
        	for(int j = i + 1;j < nums.length - 1 && nums[i] > 0;j++){
        		while(k < nums.length && nums[i] + nums[j] > nums[k])
        			k++;
        		res += k - j - 1;
        	}
        }
        return res;
    }
    
 * 2017/12/19
 * 714. Best Time to Buy and Sell Stock with Transaction Fee
 * 
    public int maxProfit(int[] prices, int fee) {
        int[] buy = new int[prices.length];
        int[] sell = new int[prices.length];
        buy[0] -= prices[0];
        sell[0] = 0;
        for(int i = 1;i < prices.length;i++){
        	buy[i] = Math.max(sell[i - 1] - prices[i], buy[i - 1]);
        	sell[i] = Math.max(sell[i - 1], buy[i - 1] + prices[i] - 2);
        }
        return Math.max(buy[prices.length - 1], sell[prices.length - 1]);
    }
    
 * 2017/12/19
 * 62. Unique Paths
 * 
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for(int i = 0;i < m;i++){
        	dp[i][0] = 1;
        }
        for(int i = 0;i < n;i++){
        	dp[0][i] = 1;
        }
        for(int i = 1;i < m;i++){
        	for(int j = 1;j < n;j++){
        		dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        	}
        }
        return dp[m - 1][n - 1];
    }
    
 * 2017/12/19
 * 386. Lexicographical Numbers
 * 
class Solution {
    public List<Integer> lexicalOrder(int n) {
        List<Integer> res = new ArrayList<>();
        for(int i = 1;i <= 9;i++){
        	if(i <= n){
        		res.add(i);
        		lexicalPutInArray(n, i, res);
        	}
        }
        return res;
    }
    
    public void lexicalPutInArray(int n,int t,List<Integer> res){
    	for(int i = 0;i <= 9;i++){
    		int temp = t * 10 + i;
    		if(temp <= n){
    			res.add(temp);
    			lexicalPutInArray(n, temp, res);
    		}
    		else{
    			return;
    		}
    	}
    }
}

 * 2017/12/19
 * 394. Decode String
 * 
    public String decodeString(String s) {
        Stack<Integer> stack = new Stack<Integer>();
        for(int i = 0;i < s.length();i++){
        	if(s.charAt(i) == '['){
        		stack.add(i);
        	}
        	if(s.charAt(i) == ']'){
        		int left = stack.pop();
        		int right = i;
        		String temp = s.substring(left + 1,right);
        		int num = left - 1;
        		while(num >= 0 && s.charAt(num) >= '0' && s.charAt(num) <= '9')num--;
        		num++;
        		int times = Integer.parseInt(s.substring(num,left));
        		i = num + times * temp.length() - 1;
        		StringBuffer buffer = new StringBuffer().append(s.substring(0,num));
        		while(times > 0){
        			buffer.append(temp);
        			times--;
        		}
        		buffer.append(s.substring(right + 1));
        		s = buffer.toString();
        	}
        }
        return s;
    }
 */

/**
 * 2017/12/20
 * 89. Gray Code
 * 
    public List<Integer> grayCode(int n) {
        List<Integer> list = new ArrayList<>();
        if(n < 0)return list;
        if(n == 0){
        	list.add(0);
        	return list;
        }
        list.add(0);
        list.add(1);
        int num = list.size();
        for(int i = 1;i < n;i++){
        	num = list.size();
        	for(int j = num - 1;j >= 0;j--){
        		list.add(num + list.get(j));
        	}
        }
        return list;
    }
 */

/**
 * 2017/12/22
 * 199. Binary Tree Right Side View
 * 
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if(root == null)return list;
        queue.add(root);
        while(!queue.isEmpty()){
        	int num = queue.size();
        	for(int i = 0;i < num - 1;i++){
        		root = queue.poll();
        		if(root.left != null)queue.add(root.left);
        		if(root.right != null)queue.add(root.right);
        	}
        	root = queue.poll();
        	list.add(root.val);
    		if(root.left != null)queue.add(root.left);
    		if(root.right != null)queue.add(root.right);
        }
        return list;
    }
 */

/**
 * 2017/12/23
 * 740. Delete and Earn
 * 
    public int deleteAndEarn(int[] nums) {
    	//对nums中的数据整理成数组list[2],list[0]代表数字list[1]代表该数字总和
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0;i < nums.length;i++){
        	if(map.containsKey(nums[i])){
        		map.put(nums[i], map.get(nums[i]) + nums[i]);
        	}
        	else{
        		map.put(nums[i], nums[i]);
        	}
        }
        List<int[]> list = new ArrayList<>();
        for(int key : map.keySet()){
        	int[] temp = new int[2];
        	temp[0] = key;
        	temp[1] = map.get(key);
        	list.add(temp);
        }
        //对数组按升序排列
        Collections.sort(list,new Comparator<int[]>() {
            public int compare(int[] o1, int[] o2) {  
                return o1[0] - o2[0];
            }  
		});
        if(list.size() == 0)return 0;
        if(list.size() == 1)return list.get(0)[1];
        int[] dp = new int [list.size()];
        //动态规划解决问题
        dp[0] = list.get(0)[1];
        if(list.get(1)[0] == list.get(0)[0] + 1){
        	dp[1] = Math.max(list.get(1)[1], list.get(0)[1]);
        }
        else{
        	dp[1] = list.get(1)[1] + list.get(0)[1];
        }
        for(int i = 2;i < list.size();i++){
        	if(list.get(i)[0] == list.get(i - 1)[0] + 1){
        		dp[i] = Math.max(dp[i - 1], dp[i - 2] + list.get(i)[1]);
        	}
        	else{
        		dp[i] = dp[i - 1] + list.get(i)[1];
        	}
        }
        return dp[list.size() - 1];
    }
    
 * 2017/12/23
 * 96. Unique Binary Search Trees
 * 
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 1;
        for(int i = 2;i <= n;i++){
        	for(int j = 1;j <= i;j++){
        		dp[i] += dp[j - 1] * dp[i - j];
        	}
        }
        return dp[n];
    }
 */

/**
 * 2017/12/24
 * 137. Single Number II
 * 
    public int singleNumber(int[] nums) {
        int x1 = 0,x2 = 0,mask = 0;
        for(int i : nums){
            x2 ^= (x1 & i);
            x1 ^= i;
        	mask = ~(x1 & x2);
        	x1 &= mask;
        	x2 &= mask;
        }
        return x1;
    }
 */

/**
 * 2017/12/25
 * 738. Monotone Increasing Digits
 * 
class Solution {
    public int monotoneIncreasingDigits(int N) {
    	int temp = N;//储存N的值
        Stack<Integer> stack = new Stack<>();//储存N的每位数据
        while(temp > 0){
        	stack.add(temp % 10);
        	temp = temp / 10;
        }
        int[] bits = new int[stack.size()];
        int i = 0;
        while(!stack.isEmpty()){
        	bits[i++] = stack.pop();
        }
        for(i = 1;i < bits.length;i++){
        	if(bits[i] < bits[i - 1])break;//找到终止递增的位
        }
        if(i < bits.length){
        	int res = bitArrayMinusOne(bits, i);//位于终止位前的数字需减一
        	for(int j = 0;j < bits.length - i;j++){
        		res = res * 10 + 9;
        	}
        	return res;
        }
        return N;
    }
    
    public int bitArrayMinusOne(int[] bits,int length){
    	if(length == 1)return bits[0] - 1;
    	int i;
    	for(i = length - 1;i > 0;i--){
    		if(bits[i] >= bits[i - 1] + 1){
    			bits[i]--;
    			break;
    		}
    	}
    	if(i == 0)bits[0]--;
    	int res = 0;
    	for(int j = 0;j <= i;j++){
    		res = res * 10 + bits[j];
    	}
    	for(int j = 0;j < length - i - 1;j++){
    		res = res * 10 + 9;
    	}
    	return res;
    }
}

 * 2017/12/25
 * 399. Evaluate Division
 * 
class Solution {
    public double[] calcEquation(String[][] equations, double[] values, String[][] queries) {
        double[] res = new double[queries.length];
        Map<String, Map<String,Double>> map = new HashMap<>();
        for(int i = 0;i < equations.length;i++){//初始化图
        	if(!map.containsKey(equations[i][0])){
        		Map<String, Double> temp = new HashMap<>();
        		temp.put(equations[i][1], values[i]);
        		map.put(equations[i][0], temp);
        	}
        	else{
        		Map<String, Double> temp = map.get(equations[i][0]);
        		temp.put(equations[i][1], values[i]);
        		map.put(equations[i][0], temp);
        	}
        	if(!map.containsKey(equations[i][1])){
        		Map<String, Double> temp = new HashMap<>();
        		temp.put(equations[i][0], 1 / values[i]);
        		map.put(equations[i][1], temp);
        	}
        	else{
        		Map<String, Double> temp = map.get(equations[i][1]);
        		temp.put(equations[i][0], 1 / values[i]);
        		map.put(equations[i][1], temp);
        	}
        }
        for(int i = 0;i < res.length;i++){
        	res[i] = singleQueryValue(map, queries[i][0], queries[i][1], new HashSet<String>());
        }
        return res;
    }
    
    public double singleQueryValue(Map<String, Map<String,Double>> map, String a,String b,Set<String> set){
    	//a,b分别为分子分母,set为已经遍历的字符串
    	if(map.containsKey(a) && map.get(a).containsKey(b)){
    		return map.get(a).get(b);
    	}
        if(!map.containsKey(a))return -1.0;
    	Map<String,Double> temp = map.get(a);
    	for(String str : temp.keySet()){
    		if(!set.contains(str)){
    			set.add(str);
    			double middle = singleQueryValue(map, str, b, set);
    			if(middle != -1.0)return map.get(a).get(str) * middle;
    			set.remove(str);
    		}
    	}
    	return -1.0;
    }
}

 * 2017/12/25
 * 436. Find Right Interval
 * 
class Solution {
	public class Interval {
		int start;
		int end;
		Interval() {
			start = 0;
			end = 0;
		}
		Interval(int s, int e) {
			start = s;
			end = e;
		}
	}
	
   public int[] findRightInterval(Interval[] intervals) {
       int[] res = new int[intervals.length];
       int[] left = new int[intervals.length];
       Map<Integer, Integer> map = new HashMap<>();//储存间隔左值所对应的下标
		for (int i = 0; i < intervals.length; i++) {
			left[i] = intervals[i].start;
			map.put(intervals[i].start, i);
		}
		Arrays.sort(left);
		for(int i = 0;i < intervals.length;i++){
			int index = Arrays.binarySearch(left, intervals[i].end);
			if(index == -1 - left.length)res[i] = -1;
			else if(index >= 0)res[i] = map.get(left[index]);
			else res[i] = map.get(left[-(1 + index)]);
		}
		return res;
	}
}

 * 2017/12/25
 * 744. Find Smallest Letter Greater Than Target
 * 
    public char nextGreatestLetter(char[] letters, char target) {
        int index = Arrays.binarySearch(letters, target);
        if(index >= 0){
        	while(index < letters.length && letters[index] == target){
        		index++;
        	}
        	if(index == letters.length)return letters[0];
        	else return letters[index];
        }
        else if(index == -1 - letters.length){
        	return letters[0];
        }
        else{
        	return letters[-1 - index];
        }
    }
 */

/**
 * 2017/12/26
 * 102. Binary Tree Level Order Traversal
 * 
    public List<List<Integer>> levelOrder(TreeNode root) {
    	List<List<Integer>> res = new ArrayList<>();
    	Queue<TreeNode> queue = new LinkedList<>();
    	TreeNode temp = root;
    	if(temp == null)return res;
    	queue.add(temp);
    	while(!queue.isEmpty()){
    		int num = queue.size();
    		List<Integer> list = new ArrayList<>();
    		for(int i = 0;i < num;i++){
    			temp = queue.poll();
    			list.add(temp.val);
    			if(temp.left != null)queue.add(temp.left);
    			if(temp.right != null)queue.add(temp.right);
    		}
    		res.add(list);
    	}
    	return res;
    }
    
 * 2017/12/26
 * 525. Contiguous Array
 * 
    public int findMaxLength(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        int max = 0, count = 0;
        map.put(0, -1);
        for(int i = 0;i < nums.length;i++){
        	count += nums[i] == 0? -1 : 1;
        	if(map.containsKey(count)){
        		max = Math.max(max, i - map.get(count));
        	}
        	else{
        		map.put(count, i);
        	}
        }
        return max;
    }
    
 * 2017/12/26
 * 692. Top K Frequent Words
 * 
    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> map = new HashMap<>();
        for(String word : words){
        	map.put(word, map.getOrDefault(word, 0) + 1);
        }
        List<Map.Entry<String, Integer>> list = new ArrayList<>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<String,Integer>>(){
			@Override
			public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
				// TODO 自动生成的方法存根
				if(o1.getValue() == o2.getValue()){
					return o1.getKey().compareTo(o2.getKey());
				}
				return o2.getValue() - o1.getValue();
			}
        });
        List<String> res = new ArrayList<>();
        for(int i = 0;i < k && i < list.size();i++){
        	res.add(list.get(i).getKey());
        }
        return res;
    }
    
 * 2017/12/26
 * 77. Combinations
 * 
class Solution {
    public List<List<Integer>> combine(int n, int k) {
    	if(n == k){
			List<List<Integer>> list = new ArrayList<>();
			List<Integer> temp = new ArrayList<>();
			for (int i = 1; i <= n; i++) {
				temp.add(i);
			}
			list.add(temp);
			return list;
		}
    	List<List<Integer>> res = new ArrayList<>();
        combine(res, new ArrayList<Integer>(), 1, n, k);
        return res;
    }
    
    public void combine(List<List<Integer>> res, List<Integer> list, int start,int n,int k){
    	if(k == 0){
    		res.add(new ArrayList<>(list));
    		return;
    	}
    	for(int i = start;i <= n;i++){
    		list.add(i);
    		combine(res, list, i + 1, n, k - 1);
    		list.remove(list.size() - 1);
    	}
    }
}
 */


/**
 * 2017/12/27
 * 560. Subarray Sum Equals K
 * 
    public int subarraySum(int[] nums, int k) {
        int res = 0;
        int count = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for(int num : nums){
        	count += num;
        	if(map.containsKey(count - k))res += map.get(count - k);
        	map.put(count, map.getOrDefault(count, 0) + 1);
        }	
        return res;
    }
    
 * 2017/12/27
 * 718. Maximum Length of Repeated Subarray
 * 
     public int findLength(int[] A, int[] B) {
        int m = A.length,n = B.length;
        if(m == 0 || n == 0)return 0;
        int max = 0;
        int[] preDP = new int[n];
        for(int i = 0;i < n;i++){
        	if(A[0] == B[i])preDP[i] = 1;
        }
        for(int i = 1;i < m;i++){
        	int[] dp = new int[n];
        	if(B[0] == A[i])dp[0] = 1;
        	for(int j = 1;j < n;j++){
        		if(A[i] == B[j]){
        			dp[j] = preDP[j - 1] + 1;
        			max = Math.max(max, dp[j]);
        		}
        	}
        	preDP = dp;
        }
        return max;
    }
    
 * 2017/12/27
 * 746. Min Cost Climbing Stairs
 * 
    public int minCostClimbingStairs(int[] cost) {
        if(cost.length == 2){
        	return Math.min(cost[0], cost[1]);
        }
        if(cost.length == 1)return cost[0];
        if(cost.length == 0)return 0;
    	int[] dp = new int[cost.length];
        dp[0] = cost[0];
        dp[1] = cost[1];
        for(int i = 2;i < cost.length;i++){
        	dp[i] = Math.min(dp[i - 1], dp[i - 2]) + cost[i];
        }
        return Math.min(dp[dp.length - 1], dp[dp.length - 2]);
    }
    
 * 2017/12/27
 * 482. License Key Formatting
 * 
    public String licenseKeyFormatting(String S, int K) {
        String[] tempSplit = S.split("-");
        StringBuffer res = new StringBuffer();
        StringBuffer temp = new StringBuffer();
        for(String str : tempSplit){
        	temp.append(str);
        }
        int c = temp.length() % K;
        res.append(temp.substring(0,c));
        int time = temp.length() / K;
        for(int i = 0;i < time;i++){
        	res.append("-").append(temp.substring(c + i * K, c + (i + 1) * K));
        }
        if(c == 0){
        	if(res.length() >= K)return res.substring(1).toUpperCase();
        	return res.toString().toUpperCase();
        }
        return res.toString().toUpperCase();
    }
    
 * 2017/12/27
 * 748. Shortest Completing Word
 * 
class Solution {
    public String shortestCompletingWord(String licensePlate, String[] words) {
        Map<Character, Integer> map = new HashMap<>();
        for(char license : licensePlate.toUpperCase().toCharArray()){
        	if(license >= 'A' && license <= 'Z'){
        		map.put(license, map.getOrDefault(license, 0) + 1);
        	}
        }
        int minLength = Integer.MAX_VALUE;
        String res = new String();
        for(String word : words){
        	if(isCompletingWord(map, word.toUpperCase()) && word.length() < minLength){
                minLength = word.length();
        		res = word;
        	}
        }
        return res;
    }
    
    public boolean isCompletingWord(Map<Character, Integer> map,String words){
    	Map<Character,Integer> temp = new HashMap<>();
        for(char word : words.toUpperCase().toCharArray()){
        	if(word >= 'A' && word <= 'Z'){
        		temp.put(word, temp.getOrDefault(word, 0) + 1);
        	}
        }
        for(char flag : map.keySet()){
        	if(!temp.containsKey(flag) || temp.get(flag) < map.get(flag)){
        		return false;
        	}
        }
        return true;
    }
}

 * 2017/12/27
 * 747. Largest Number At Least Twice of Others
 * 
    public int dominantIndex(int[] nums) {
    	if(nums.length == 0)return -1;
    	if(nums.length == 1)return 0;
        int max = nums[0];
        int secMax = Integer.MIN_VALUE;
        int index = 0;
        for(int i = 1;i < nums.length;i++){
        	if(nums[i] > max){
        		secMax = max;
        		max = nums[i];
        		index = i;
        	}
        	else if(nums[i] > secMax){
        		secMax = nums[i];
        	}
        }
        if(max > 2 * secMax)return index;
        return -1;
    }
 */

/**
 * 2017/12/28
 * 59. Spiral Matrix II
 * 
    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        int left = 0,right = n - 1;
        int top = 0,down = n - 1;
        int count = 1;
        while(left <= right){
        	for(int i = left;i <= right;i++){
        		res[top][i] = count++;
        	}
        	top++;
        	for(int i = top;i <= down;i++){
        		res[i][right] = count++;
        	}
        	right--;
        	for(int i = right;i >= left;i--){
        		res[down][i] = count++;
        	}
        	down--;
        	for(int i = down;i >= top;i--){
        		res[i][left] = count++;
        	}
        	left++;
        }
        return res;
    }
    
 * 2017/12/28
 * 48. Rotate Image
 * 
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        int left = 0,right = n - 1;
        int top = 0,down = n - 1;
        while(left < right){
        	for(int i = 0;i < right - left;i++){
        		int temp = matrix[top][left + i];
        		matrix[top][left + i] = matrix[down - i][left];
        		matrix[down - i][left] = matrix[down][right - i];
        		matrix[down][right - i] = matrix[top + i][right];
        		matrix[top + i][right] = temp;
        	}
        	top++;
        	down--;
        	left++;
        	right--;
        }
    }
    
 * 2017/12/28
 * 39. Combination Sum
 * 
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        combination(res, new ArrayList<>(), candidates, 0, target);
        return res;
    }
    
    void combination(List<List<Integer>> res,List<Integer> sub,int[] candidates,int start,int target){
    	if(target == 0){
    		res.add(sub);
    		return;
    	}
    	if(start == candidates.length)return;
		combination(res, new ArrayList<>(sub), candidates, start + 1, target);
    	for(int i = 1;i <= target / candidates[start];i++){
    		sub.add(candidates[start]);
    		combination(res, new ArrayList<>(sub), candidates, start + 1, target - i * candidates[start]);
    	}
    }
}

 * 2017/12/28
 * 593. Valid Square
 * 
class Solution {
    public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
    	long[] lens = {length(p1, p2),length(p1, p3),length(p1, p4),length(p3, p2),length(p4, p2),length(p3, p4)};
    	long nonMax = 0;
    	long max = 0;
    	int count = 6;
    	for(long len : lens){
    		max = Math.max(max, len);
    	}
    	for(long len:lens){
    		if(len != max){
    			nonMax = len;
    			count--;
    		}
    	}
    	if(count != 2)return false;
    	for(long len : lens){
    		if(len != max && len != nonMax)return false;
    	}
    	return true;
    }
    
    public long length(int[] p1,int p2[]){
    	return (long)Math.pow(p1[0] - p2[0], 2) + (long)Math.pow(p1[1] - p2[1], 2);
    }
}
 */

/**
 * 2017/12/29
 * 153. Find Minimum in Rotated Sorted Array
 * 
    public int findMin(int[] nums) {
        int left = 0,right = nums.length - 1;
        while(left < right){
        	int mid = (left + right) / 2;
        	if(nums[mid] > nums[right]){
        		left = mid + 1;
        	}
        	else if(nums[mid] < nums[right]){
        		right = mid;
        	}
        }
        return nums[left];
    }
    
 * 2017/12/29
 * 684. Redundant Connection
 * 
class Solution {
    public int[] findRedundantConnection(int[][] edges) {
    	int[] parent = new int[edges.length + 1];
    	for(int i = 1;i < edges.length + 1;i++){
    		parent[i] = i;
    	}
        for(int[] edge : edges){
        	int left = find(parent, edge[0]);
        	int right = find(parent, edge[1]);
        	if(left == right)return edge;
        	parent[left] = right;
        }
        return new int[2];
    }
    
    private int find(int[] parent,int p){
    	while(p != parent[p]){
    		parent[p] = parent[parent[p]];
    		p = parent[p];
    	}
    	return p;
    }
}

 * 2017/12/29
 * 435. Non-overlapping Intervals
 * 
class Solution {
	public class Interval {
		int start;
		int end;
		Interval() {
			start = 0;
			end = 0;
		}
		Interval(int s, int e) {
			start = s;
			end = e;
		}
	}
	
    public int eraseOverlapIntervals(Interval[] intervals) {
        if(intervals.length == 0)return 0;
        Arrays.sort(intervals, new Comparator<Interval>(){
			public int compare(Interval o1, Interval o2) {
				return o1.end - o2.end;
			}
        });
        int end = intervals[0].end;
        int count = 1;
        for(int i = 1;i < intervals.length;i++){
        	if(intervals[i].start >= end){
        		count++;
        		end = intervals[i].end;
        	}
        }
        return intervals.length - count;
    }
}
 */

/**
 * 2017/12/30
 * 64. Minimum Path Sum
 * 
    public int minPathSum(int[][] grid) {
    	if(grid.length == 0 || grid[0].length == 0)return 0;
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for(int i = 1;i < n;i++){
        	dp[0][i] = dp[0][i - 1] + grid[0][i];
        }
        for(int i = 1;i < m;i++){
        	dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for(int i = 1;i < m;i++){
        	for(int j = 1;j < n;j++){
        		dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
        	}
        }
        return dp[m - 1][n - 1];
    }
    
 * 2017/12/30
 * 300. Longest Increasing Subsequence
 * 
    public int lengthOfLIS(int[] nums) {
    	if(nums.length == 0)return 0;
    	if(nums.length == 1)return 1;
        int[][] dp = new int[nums.length][2];
        int res = 1;
        dp[0][0] = nums[0];
        dp[0][1] = 1;
        for(int i = 1;i < nums.length;i++){
        	int max = 0;
        	for(int j = 0;j < i;j++){
        		if(nums[i] > dp[j][0]){
        			max = Math.max(max, dp[j][1]);
        		}
        	}
        	dp[i][0] = nums[i];
        	dp[i][1] = max + 1;
        	res = Math.max(res, dp[i][1]);
        }
        return res;
    }
    
 * 2017/12/30
 * 334. Increasing Triplet Subsequence
 * 
    public boolean increasingTriplet(int[] nums) {
    	int[] tail = new int[3];
    	int size = 0;
    	for(int num : nums){
    		int left = 0,right = size;
    		while(left != right){
    			int mid = (left + right) / 2;
    			if(tail[mid] >= num){
    				right = mid;
    			}
    			else{
    				left = mid + 1;
    			}
    		}
    		if(left == 2)return true;
    		tail[left] = num;
    		if(left == size)size++;
    	}
    	return false;
    }
    
 * 2017/12/30
 * 491. Increasing Subsequences
 * 
class Solution {
    public List<List<Integer>> findSubsequences(int[] nums) {
        Set<List<Integer>> res = new HashSet<>();
        List<Integer> list = new ArrayList<>();
        subsequences(res, list, nums, 0);
        return new ArrayList<>(res);
    }
    
    public void subsequences(Set<List<Integer>> res,List<Integer> list, int[] nums,int start){
    	if(list.size() >= 2){
    		List<Integer> temp = new ArrayList<>(list);
    		res.add(temp);
    	}
    	for(int i = start;i < nums.length;i++){
    		if(list.size() == 0 || list.get(list.size() - 1) <= nums[i]){
    			list.add(nums[i]);
    			subsequences(res, list, nums, i + 1);
    			list.remove(list.size() - 1);
    		}
    	}
    }
}

 * 2017/12/30
 * 240. Search a 2D Matrix II
 * 
    public boolean searchMatrix(int[][] matrix, int target) {
    	if(matrix.length == 0 || matrix[0].length == 0)return false;
        int m = matrix.length;
    	int n = matrix[0].length;
    	int[] column = new int[m];
    	for(int i = 0;i < m;i++){
    		column[i] = matrix[i][n - 1];
    	}
    	//采用二分搜索比较数组end与target值，返回top行
    	int top = Arrays.binarySearch(column, target);
    	if(top >= 0)return true;
    	top = - 1 - top;
    	if(top == m)return false;
    	for(int i = 0;i < m;i++){
    		column[i] = matrix[i][0];
    	}
    	//采用二分搜索比较数组start与target值，返回down行
    	int down = Arrays.binarySearch(column, target);
    	if(down >= 0)return true;
    	down = -1 - down;
    	//target在top行与down行之间
    	for(int i = top;i < down;i++){
    		if(Arrays.binarySearch(matrix[i], target) >= 0)return true;
    	}
    	return false;
    }
 */
package exercise;

class Solution_Dec19th_To_Dec30th {

}












