/**
 * @author jhy code from 4.24 to 5.6
 * 29 questions
 */

/**
 * 2018/4/24
 * 115. Distinct Subsequences
 * 
    public int numDistinct(String s, String t) {
        int[][] dp = new int[t.length() + 1][s.length() + 1];
        for(int i = 0;i < s.length() + 1;i++){
        	dp[0][i] = 1;
        }
        for(int i = 0;i < t.length();i++){
        	for(int j = 0;j < s.length();j++){
        		if(t.charAt(i) == s.charAt(j)){
        			dp[i + 1][j + 1] = dp[i][j] + dp[i + 1][j];
        		}else{
        			dp[i + 1][j + 1] = dp[i + 1][j];
        		}
        	}
        }
        return dp[t.length()][s.length()];
    }
    
 * 2018/4/24
 * 552. Student Attendance Record II
 * 
    public int checkRecord(int n) {
        final int MOD = 1000000007;
        int[][][] dp = new int[n + 1][2][3];//[i][j][k], j->num of A, k->num of L
        dp[0] = new int[][]{{1, 1, 1}, {1, 1, 1}};
        for(int i = 1;i < n + 1;i++){
        	for(int j = 0;j < 2;j++){
        		for(int k = 0;k < 3;k++){
        			int val = dp[i - 1][j][2];
        			if(j > 0)val = (val + dp[i - 1][j - 1][2]) % MOD;
        			if(k > 0)val = (val + dp[i - 1][j][k - 1]) % MOD;
        			dp[i][j][k] = val;
        		}
        	}
        }
        return dp[n][1][2];
    }
    
 * 2018/4/24
 * 600. Non-negative Integers without Consecutive Ones
 * 
    public int findIntegers(int num) {
        int[] bit = new int[32];
        bit[0] = 1;
        bit[1] = 2;
        for(int i = 2;i < bit.length;i++){
        	bit[i] = bit[i - 1] + bit[i - 2];
        }
        int i = 30, sum = 0, pre_bit = 0;
        while(i >= 0){
        	if((num & (1 << i)) != 0){
        		sum += bit[i];
        		if(pre_bit == 1){
        			sum--;
        			break;
        		}
        		pre_bit = 1;
        	}else{
        		pre_bit = 0;
        	}
        	i--;
        }
        return sum + 1;
    }
    
 * 2018/4/24
 * 564. Find the Closest Palindrome
 * 
class Solution {
    public String nearestPalindromic(String n) {
        Long num = Long.parseLong(n);
        Long high = findHigherPalindrome(num + 1);
        Long low = findLowerPalindrome(num - 1);
        return Math.abs(num - low) > Math.abs(num - high) ? String.valueOf(high) : String.valueOf(low);
    }
    
    public Long findHigherPalindrome(Long limit){
    	char[] origin = Long.toString(limit).toCharArray();
    	int m = origin.length;
    	char[] target = Arrays.copyOf(origin, m);
    	for(int i = 0;i < m / 2;i++){
    		target[m - 1 - i] = target[i];
    	}
    	for(int i = 0;i < m;i++){
    		if(origin[i] < target[i]){
    			return Long.parseLong(String.valueOf(target));
    		}else if(origin[i] > target[i]){
    			for(int j = (m - 1) / 2;j >= 0;j--){
    				if(++target[j] > '9'){
    					target[j] = '0';
    				}else{
    					break;
    				}
    			}
    			for(int k = 0;k < m / 2;k++)target[m - 1 - k] = target[k];
    			return Long.parseLong(String.valueOf(target));
    		}
    	}
    	return Long.parseLong(String.valueOf(target));
    }
    
    public Long findLowerPalindrome(Long limit){
    	char[] origin = Long.toString(limit).toCharArray();
    	int m = origin.length;
    	char[] target = Arrays.copyOf(origin, m);
    	for(int i = 0;i < m / 2;i++){
    		target[m - 1 - i] = target[i];
    	}
    	for(int i = 0;i < m;i++){
    		if(origin[i] > target[i]){
    			return Long.parseLong(String.valueOf(target));
    		}else if(origin[i] < target[i]){
    			for(int j = (m - 1) / 2;j >= 0;j--){
    				if(--target[j] < '0'){
    					target[j] = '9';
    				}else{
    					break;
    				}
    			}
    			if(target[0] == '0'){
    				char[] temp = new char[m - 1];
    				Arrays.fill(temp, '9');
    				return Long.parseLong(String.valueOf(temp));
    			}
    			for(int k = 0;k < m / 2;k++)target[m - 1 - k] = target[k];
    			return Long.parseLong(String.valueOf(target));
    		}
    	}
    	return Long.parseLong(String.valueOf(target));
    }
}
*/

/**
 * 2018/4/25
 * 25. Reverse Nodes in k-Group
 * 
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        int count = k;
        ListNode pre = null;
        ListNode cur = head;
        ListNode tail = null;
        ListNode tempHead = null;
        while(cur != null && count-- > 0){
        	pre = cur;
        	cur = cur.next;
        }
        if(count > 0)return head;
        tempHead = pre.next;
        pre.next = null;
        ListNode[] res = reverse(head);
        head = res[0];
        tail = res[1];
        while(cur != null){
        	count = k;
        	while(cur != null && count-- > 0){
            	pre = cur;
            	cur = cur.next;
            }
        	if(count > 0){
        		tail.next = tempHead;
        		break;
        	}
        	ListNode temp = tempHead;
        	tempHead = pre.next;
        	pre.next = null;
        	res = reverse(temp);
        	tail.next = res[0];
        	tail = res[1];
        }
        return head;
    }
    
    public ListNode[] reverse(ListNode head){//翻转链表
    	ListNode cur = head;
    	ListNode pre = null;
    	ListNode post = null;
    	while(cur != null){
    		post = cur.next;
    		cur.next = pre;
    		pre = cur;
    		cur = post;
    	}
    	return new ListNode[]{pre, head};//返回链表头和尾
    }
}

 * 2018/4/25
 * 99. Recover Binary Search Tree
 * 
class Solution {
	TreeNode pre = new TreeNode(Integer.MIN_VALUE);
	TreeNode first = null;
	TreeNode second = null;
	
    public void recoverTree(TreeNode root) {
        traverse(root);
        int temp = first.val;
        first.val = second.val;
        second.val = temp;
    }
    
    public void traverse(TreeNode root){
    	if(root == null)return;
    	traverse(root.left);
    	if(first == null && pre.val >= root.val){
    		first = pre;
    	}
    	if(first != null && pre.val >= root.val){
    		second = root;
    	}
    	pre = root;
    	traverse(root.right);
    }
}

 * 2018/4/25
 * 37. Sudoku Solver
 * 
class Solution {
    public void solveSudoku(char[][] board) {
        if(board == null || board.length == 0)return;
        help(board);
    }
    
    public boolean help(char[][] board){
    	for(int i = 0;i < 9;i++){
    		for(int j = 0;j < 9;j++){
    			if(board[i][j] == '.'){
    				for(char k = '1';k <= '9';k++){
    					if(isValid(board, i, j, k)){
    						board[i][j] = k;
    						if(help(board))return true;
    						board[i][j] = '.';
    					}
    				}
    				return false;
    			}
    		}
    	}
    	return true;
    }
    
    public boolean isValid(char[][] board, int row, int col, char c){
    	for(int i = 0;i < 9;i++){
    		if(board[i][col] == c)return false;
    		if(board[row][i] == c)return false;
    		if(board[row / 3 * 3 + i / 3][col / 3 * 3 + i % 3] == c)return false;
    	}
    	return true;
    }
}

 * 2018/4/25
 * 32. Longest Valid Parentheses
 * 
    public int longestValidParentheses(String s) {
        int max = 0;
        int[] dp = new int[s.length()];
        for(int i = 1;i < s.length();i++){
        	if(s.charAt(i) == ')'){
        		if(s.charAt(i - 1) == '('){
        			dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
        		}else if(i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '('){
        			dp[i] = (i - dp[i - 1] >= 2 ? dp[i - dp[i - 1] - 2] : 0) + dp[i - 1] + 2;
        		}
        		max = Math.max(max, dp[i]);
        	}
        }
        return max;
    }
    
 * 2018/4/25
 * 273. Integer to English Words
 * 
class Solution {
    private final String[] belowTen = new String[] {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
    private final String[] belowTwenty = new String[] {"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
    private final String[] belowHundred = new String[] {"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
    
    public String numberToWords(int num) {
        if(num == 0)return "Zero";
        return helper(num);
    }
    
    public String helper(int num){
    	String result = new String();
        if (num < 10) result = belowTen[num];
        else if (num < 20) result = belowTwenty[num -10];
        else if (num < 100) result = belowHundred[num/10] + " " + helper(num % 10);
        else if (num < 1000) result = helper(num/100) + " Hundred " +  helper(num % 100);
        else if (num < 1000000) result = helper(num/1000) + " Thousand " +  helper(num % 1000);
        else if (num < 1000000000) result = helper(num/1000000) + " Million " +  helper(num % 1000000);
        else result = helper(num/1000000000) + " Billion " + helper(num % 1000000000);
        return result.trim();
    }
}
*/

/**
 * 2018/4/26
 * 72. Edit Distance
 * 
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for(int i = 0;i <= m;i++)dp[i][0] = i;
        for(int i = 0;i <= n;i++)dp[0][i] = i;
        for(int i = 0;i < m;i++){
        	for(int j = 0;j < n;j++){
        		if(word1.charAt(i) == word2.charAt(j)){
        			dp[i + 1][j + 1] = dp[i][j];
        		}else{
        			dp[i + 1][j + 1] = Math.min(dp[i][j + 1], Math.min(dp[i + 1][j], dp[i][j])) + 1;
        		}
        	}
        }
        return dp[m][n];
    }
    
 * 2018/4/26
 * 164. Maximum Gap
 * 
    public int maximumGap(int[] nums) {
        if(nums == null || nums.length < 2)return 0;
        int max = 0;
        for(int num : nums)max = Math.max(max, num);
        int exp = 1;
        int[] AUX = new int[nums.length];
        while(max / exp > 0){
        	int[] count = new int[10];
        	for(int num : nums){
        		count[(num / exp) % 10]++;
        	}
        	for(int i = 1;i < 10;i++)count[i] += count[i - 1];
        	for(int i = nums.length - 1;i >= 0;i--){
        		AUX[--count[(nums[i] / exp) % 10]] = nums[i];
        	}
        	for(int i = 0;i < nums.length;i++)nums[i] = AUX[i];
        	exp *= 10;
        }
        int res = 0;
        for(int i = 1;i < nums.length;i++){
        	res = Math.max(res, nums[i] - nums[i - 1]);
        }
        return res;
    }
    
 * 2018/4/26
 * 480. Sliding Window Median
 * 
class Solution {
	public double[] medianSlidingWindow(int[] nums, int k) {
		List<Integer> window = new ArrayList<Integer>();
		double[] res = new double[nums.length - k + 1];
		for (int i = 0; i < nums.length; i++) {
			if (i >= k){
				res[i - k] = (k % 2 == 0 ? 
						((double) window.get(k / 2) + (double) window.get(k / 2 - 1)) / 2 
						: (double) window.get(k / 2));
			}
			window.add(find(window, nums[i]), nums[i]);
			if (i >= k)window.remove(find(window, nums[i - k]));
		}
		res[res.length - 1] = (k % 2 == 0 ? 
				((double) window.get(k / 2) + (double) window.get(k / 2 - 1)) / 2
				: (double) window.get(k / 2));
		return res;
	}

	public int find(List<Integer> list, int target) {
		int start = 0, end = list.size() - 1;
		while (start <= end) {
			int mid = start + (end - start) / 2;
			if (list.get(mid) < target) {
				start = mid + 1;
			} else if (list.get(mid) > target) {
				end = mid - 1;
			} else {
				return mid;
			}
		}
		return start;
	}
}

 * 2018/4/26
 * 123. Best Time to Buy and Sell Stock III
 * 
    public int maxProfit(int[] prices) {
        int buy1 = Integer.MIN_VALUE;
        int buy2 = Integer.MIN_VALUE;
        int sell1 = 0;
        int sell2 = 0;
        for(int i = 0;i < prices.length;i++){
        	buy1 = Math.max(buy1, -prices[i]);
        	sell1 = Math.max(sell1, buy1 + prices[i]);
        	buy2 = Math.max(buy2, sell1 - prices[i]);
        	sell2 = Math.max(sell2, buy2 + prices[i]);
        }
        return sell2;
    }
    
 * 2018/4/26
 * 85. Maximal Rectangle
 * 
    public int maximalRectangle(char[][] matrix) {
        if(matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)return 0;
        int m = matrix.length;
        int n = matrix[0].length;
        int[] left = new int[n];
        int[] right = new int[n];
        int[] height = new int[n];
        Arrays.fill(right, n);
        int res = 0;
        for(int i = 0;i < m;i++){
        	int cur_left = 0;
        	int cur_right = n;
        	for(int j = 0;j < n;j++){
        		if(matrix[i][j] == '1')height[j]++;
        		else height[j] = 0;
        	}
        	for(int j = 0;j < n;j++){
        		if(matrix[i][j] == '1')left[j] = Math.max(left[j], cur_left);
        		else{
        			left[j] = 0;
        			cur_left = j + 1;
        		}
        	}
        	for(int j = n - 1;j >= 0;j--){
        		if(matrix[i][j] == '1')right[j] = Math.min(right[j], cur_right);
        		else{
        			right[j] = n;
        			cur_right = j;
        		}
        	}
        	for(int j = 0;j < n;j++){
        		res = Math.max(res, (right[j] - left[j]) * height[j]);
        	}
        }
        return res;
    }
*/

/**
 * 2018/4/27
 * 472. Concatenated Words
 * 
class Solution {
    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        List<String> res = new ArrayList<>();
        Arrays.sort(words, (a, b) -> (a.length() - b.length()));
        Set<String> set = new HashSet<>();
        for(String word : words){
        	if(isValid(set, word))res.add(word);
        	set.add(word);
        }
        return res;
    }
    
    public boolean isValid(Set<String> set, String word){
    	if(set.isEmpty())return false;
    	boolean[] dp = new boolean[word.length() + 1];
    	dp[0] = true;
    	for(int i = 1;i < dp.length;i++){
    		for(int j  = 0;j < i;j++){
    			if(!dp[j])continue;
    			if(set.contains(word.substring(j, i))){
    				dp[i] = true;
    				break;
    			}
    		}
    	}
    	return dp[word.length()];
    }
}
*/

/**
 * 2018/4/29
 * 824. Goat Latin
 * 
    public String toGoatLatin(String S) {
        StringBuffer res = new StringBuffer();
        String[] words = S.split("\\s+");
        int count = 1;
        Set<Character> set = new HashSet<>(Arrays.asList(new Character[]{'a','e','i','o','u','A','E','I','O','U'}));
        for(String word : words){
        	if(set.contains(word.charAt(0))){
        		res.append(" ").append(word).append("ma");
        		for(int i = 0;i < count;i++)res.append('a');
        	}else{
        		res.append(" ").append(word.substring(1)).append(word.charAt(0)).append("ma");
        		for(int i = 0;i < count;i++)res.append('a');
        	}
        	count++;
        }
        return res.toString().substring(1);
    }
    
 * 2018/4/29
 * 826. Most Profit Assigning Work
 * 
class Solution {
	class Work{
		int key;
		int value;
		public Work(int k, int v){
			key = k;
			value = v;
		}
	}
	
    public int maxProfitAssignment(int[] difficulty, int[] profit, int[] worker) {
    	List<Work> jobs = new ArrayList<>();
        int N = profit.length, res = 0, i = 0, maxp = 0;
        for (int j = 0; j < N; ++j) jobs.add(new Work(difficulty[j], profit[j]));
        Collections.sort(jobs, new Comparator<Work>() {
			public int compare(Work o1, Work o2) {
				// TODO 自动生成的方法存根
				return o1.key - o2.key;
			}
        	
		});
        Arrays.sort(worker);
        for (int ability : worker) {
            while (i < N && ability >= jobs.get(i).key)
                maxp = Math.max(jobs.get(i++).value, maxp);
            res += maxp;
        }
        return res;
    }
}
*/

/**
 * 2018/5/2
 * 825. Friends Of Appropriate Ages
 * 
    public int numFriendRequests(int[] ages) {
    	//two pointer
    	if(ages == null || ages.length == 0)return 0;
        Arrays.sort(ages);
        int res = 0;
        int left = 0;
        int right = 1;
        while(left < right && left < ages.length){
        	while(right < ages.length && ages[left] > 0.5 * ages[right] + 7 && ages[left] <= ages[right]){
        		right++;
        	}
        	int num = right > left ? right - left - 1 : 0;
        	res += num; 
        	left++;
        	while(left < ages.length && ages[left] == ages[left - 1]){
        		res += num;
        		left++;
        	}
        	while(left >= right)right++;
        }
        return res;
    }
    
 * 2018/5/2
 * 827. Making A Large Island
 * 
class Solution {
	Map<Integer, Integer> map;
	int row;
	int col;
	
    public int largestIsland(int[][] grid) {
    	if(grid == null || grid.length == 0 || grid[0] == null || grid[0].length == 0)return 0;
        int res = 0;
        map = new HashMap<>();
        row = grid.length;
        col = grid[0].length;
        colorIsland(grid);
        Set<Integer> usedColor = new HashSet<>();
        int[][] dir = new int[][]{{0, 1}, {0, - 1}, {1, 0}, {-1, 0}};
        for(int i = 0;i < row;i++){
        	for(int j = 0;j < col;j++){
        		if(grid[i][j] == 0){
        			usedColor.clear();
        			int total = 1;
        			for(int[] d : dir){
        				int r = d[0] + i;
        				int c = d[1] + j;
        				if(r >= 0 && r < row && c >=0 && c < col && !usedColor.contains(grid[r][c]) && map.containsKey(grid[r][c])){
        					total += map.get(grid[r][c]);
        					usedColor.add(grid[r][c]);
        				}
        			}
        			res = Math.max(res, total);
        		}
        	}
        }
        for(int key : map.keySet()){
        	res = Math.max(res, map.get(key));
        }
        return res;
    }
    
    public void colorIsland(int[][] grid){
    	int color = 2;
    	for(int i = 0;i < row;i++){
    		for(int j = 0;j < col;j++){
    			if(grid[i][j] == 1){
    				int count = paint(grid, i, j, color);
    				map.put(color, count);
    				color++;
    			}
    		}
    	}
    }
    
    public int paint(int[][] grid, int i, int j, int color){
    	if(i < 0 || j < 0 || i >= row || j >= col || grid[i][j] != 1)return 0;
    	grid[i][j] = color;
    	return 1 + paint(grid, i + 1, j, color) + paint(grid, i - 1, j, color) + paint(grid, i, j + 1, color) + paint(grid, i, j - 1, color);
    }
}
*/

/**
 * 2018/5/3
 * 786. K-th Smallest Prime Fraction
 * 
	public int[] kthSmallestPrimeFraction(int[] A, int K) {
		//binary search
	    double l = 0, r = 1;
	    int p = 0, q = 1;
	    for (int n = A.length, cnt = 0; true; cnt = 0, p = 0) {
	        double m = (l + r) / 2;
	        
	        for (int i = 0, j = n - 1; i < n; i++) {
	            while (j >= 0 && A[i] > m * A[n - 1 - j]) j--;
	            cnt += (j + 1);
	            
	            if (j >= 0 && p * A[n - 1 - j] < q * A[i]) {
	                p = A[i];
	                q = A[n - 1 - j];
	            }
	        }
	        
	        if (cnt < K) {
	            l = m;
	        } else if (cnt > K) {
	            r = m;
	        } else {
	            return new int[] {p, q};
	        }
	    }
	}
*/

/**
 * 2018/5/4
 * 282. Expression Add Operators
 * 
class Solution {
    public List<String> addOperators(String num, int target) {
        List<String> res = new ArrayList<>();
        if(num == null || num.length() == 0)return res;
        helper(res, target, 0, num, "", 0, 0);
        return res;
    }
    
    public void helper(List<String> res, int target, long sum, String num, String curNum, int pos, long multi){
    	//sum--current total number of curNum
    	//num--original String
    	//curNum--used String
    	//multi--reserve previous result after last +
    	if(pos == num.length()){
    		if(sum == target)res.add(curNum);
    		return;
    	}
    	for(int i = pos;i < num.length();i++){
    		if(i != pos && num.charAt(pos) == '0')break;
    		String temp = num.substring(pos, i + 1);
    		long count = Long.parseLong(temp);
    		if(pos == 0){
    			helper(res, target, count, num, temp, i + 1, count);
    		}else{
    			helper(res, target, sum + count, num, curNum + "+" + temp, i + 1, count);
    			helper(res, target, sum - count, num, curNum + "-" + temp, i + 1, -count);
    			helper(res, target, sum - multi + count * multi, num, curNum + "*" + temp, i + 1, multi * count);
    		}
    	}
    }
}

 * 2018/5/4
 * 327. Count of Range Sum
 * 
class Solution {
	class TreeNode{
		long val;
		long count;
		TreeNode left;
		TreeNode right;
		public TreeNode(long v, long c) {
			val = v;
			count = c;
		}
	}
	
    public int countRangeSum(int[] nums, int lower, int upper) {
    	TreeNode root = null;
        int res = 0;
        long sum = 0;
        for(int num : nums){
        	sum += num;
        	if(lower <= sum && sum <= upper)res++;
        	res += count(root, sum - (long)upper, sum - (long)lower);
        	root = insert(root, sum);
        }
        return res;
    }
    
    public TreeNode insert(TreeNode root, long num){
    	if(root == null){
    		return new TreeNode(num, 1);
    	}else{
    		if(root.val == num)root.count++;
    		else if(root.val > num){
    			root.left = insert(root.left, num);
    		}else{
    			root.right = insert(root.right, num);
    		}
    		return root;
    	}
    }
    
    public long count(TreeNode root, long low, long high){//count the number between low and high
    	if(root == null){
    		return 0;
    	}
    	if(root.val >= low && root.val <= high){
    		return root.count + count(root.left, low, high) + count(root.right, low, high);
    	}else if(root.val < low){
    		return count(root.right, low, high);
    	}else{
    		return count(root.left, low, high);
    	}
    }
}

 * 2018/5/4
 * 316. Remove Duplicate Letters
 * 
    public String removeDuplicateLetters(String s) {
    	if(s == null || s.length() == 0)return "";
        int[] count = new int[26];
        int pos = 0;
        for(int i = 0;i < s.length();i++)count[s.charAt(i) - 'a']++;
        for(int i = 0;i < s.length();i++){
        	if(s.charAt(i) < s.charAt(pos))pos = i;
        	if(--count[s.charAt(i) - 'a'] == 0)break;
        }
        return s.charAt(pos) + removeDuplicateLetters(s.substring(pos + 1).replaceAll("" + s.charAt(pos), ""));
    }
*/

/**
 * 2018/5/5
 * 591. Tag Validator
 * 
    public boolean isValid(String code) {
        Stack<String> stack = new Stack<>();
        for(int i = 0; i < code.length();){
            if(i>0 && stack.isEmpty()) return false;
            if(code.startsWith("<![CDATA[", i)){
                int j = i+9;
                i = code.indexOf("]]>", j);
                if(i < 0) return false;
                i += 3;
            }else if(code.startsWith("</", i)){
                int j = i + 2;
                i = code.indexOf('>', j);
                if(i < 0 || i == j || i - j > 9) return false;
                for(int k = j; k < i; k++){
                    if(!Character.isUpperCase(code.charAt(k))) return false;
                }
                String s = code.substring(j, i++);
                if(stack.isEmpty() || !stack.pop().equals(s)) return false;
            }else if(code.startsWith("<", i)){
                int j = i + 1;
                i = code.indexOf('>', j);
                if(i < 0 || i == j || i - j > 9) return false;
                for(int k = j; k < i; k++){
                    if(!Character.isUpperCase(code.charAt(k))) return false;
                }
                String s = code.substring(j, i++);
                stack.push(s);
            }else{
                i++;
            }
        }
        return stack.isEmpty();
    }
    
 * 2018/5/5
 * 87. Scramble String
 * 
    public boolean isScramble(String s1, String s2) {
        if(s1.equals(s2))
        	return true;
        int len = s1.length();
        int[] count = new int[26];
        for(int i = 0;i < len;i++){
        	count[s1.charAt(i) - 'a']++;
        	count[s2.charAt(i) - 'a']--;
        }
        for(int i = 0;i < 26;i++){
        	if(count[i] != 0)
        		return false;
        }
        for(int i = 1;i < len;i++){
        	if(isScramble(s1.substring(0, i), s2.substring(0, i)) && isScramble(s1.substring(i), s2.substring(i)))
        		return true;
        	if(isScramble(s1.substring(0, i), s2.substring(len - i)) 
        			&& isScramble(s1.substring(i), s2.substring(0, len - i)))
        		return true;
        }
        return false;
    }
*/

/**
 * 2018/5/6
 * 830. Positions of Large Groups
 * 
    public List<List<Integer>> largeGroupPositions(String S) {
    	List<List<Integer>> res = new ArrayList<>();
    	int j = 0;
    	int i = 0;
    	for(i = 0;i < S.length();i++){
    		if(S.charAt(i) == S.charAt(j))
    			continue;
    		if(i - j >= 3){
    			List<Integer> temp = new ArrayList<>();
    			temp.add(j);
    			temp.add(i - 1);
    			res.add(temp);
    		}
    		j = i;
    	}
    	if(i - j >= 3){
			List<Integer> temp = new ArrayList<>();
			temp.add(j);
			temp.add(i - 1);
			res.add(temp);
		}
    	return res;
    }
    
 * 2018/5/6
 * 831. Masking Personal Information
 * 
class Solution {
    public String maskPII(String S) {
        if(S.contains("@"))
        	return mail(S);
        return phone(S);
    }
    
    public String mail(String S){
    	int index1 = S.indexOf("@");
    	int index2 = S.indexOf(".");
    	String name1 = S.substring(0, index1).toLowerCase();
    	String name2 = S.substring(index1 + 1, index2).toLowerCase();
    	String name3 = S.substring(index2 + 1).toLowerCase();
    	return name1.charAt(0) + "*****" + name1.charAt(name1.length() - 1) + "@" + name2 + "." + name3;
    }
    
    public String phone(String S){
    	StringBuilder sb = new StringBuilder();
    	for(int i = 0;i < S.length();i++){
    		if(S.charAt(i) >= '0' && S.charAt(i) <= '9'){
    			sb.append(S.charAt(i));
    		}
    	}
    	if(sb.length() == 13)
    		return "+***-***-***-" + sb.toString().substring(sb.length() - 4);
    	else if(sb.length() == 12)
    		return "+**-***-***-" + sb.toString().substring(sb.length() - 4);
    	else if(sb.length() == 11)
    		return "+*-***-***-" + sb.toString().substring(sb.length() - 4);
    	else//sb.length() == 10
    		return "***-***-" + sb.toString().substring(sb.length() - 4);
    }
}

 * 2018/5/6
 * 829. Consecutive Numbers Sum
 * 
    public int consecutiveNumbersSum(int N) {
    	int res = 0;
    	for(int i = 1;i < 2000000;i++){
    		int mid = N / i;
    		if(mid - (i - 1) / 2 < 1)break;
    		if(i % 2 == 0){
    			if((2 * mid + 1) * (i / 2) == N)res++;
    		}else{
    			if(mid * i == N)res++;
    		}
    	}
    	return res;
    }
    
 * 2018/5/6
 * 828. Unique Letter String
 * 
    public int uniqueLetterString(String S) {
        //focus on every character instead of string
    	int[][] index = new int[26][2];
    	for(int i = 0;i < 26;i++)Arrays.fill(index[i], -1);
    	int res = 0;
    	int mod = 1000000007;
    	for(int i = 0;i < S.length();i++){
    		int ch = S.charAt(i) - 'A';
    		res = (res + (i - index[ch][1]) * (index[ch][1] - index[ch][0]) % mod) % mod;
    		index[ch] = new int[]{index[ch][1], i};
    	}
    	//last index of every upper characters
    	for(int i = 0;i < 26;i++){
    		res = (res + (S.length() - index[i][1]) * (index[i][1] - index[i][0]) % mod) % mod;
    	}
    	return res;
    }
    
 * 2018/5/6
 * 630. Course Schedule III
 * 
class Solution {
	//Solution1: backtrace--TTL
    public int scheduleCourse(int[][] courses) {
        return backtrace(courses, new boolean[courses.length], 0);
    }
    
    public int backtrace(int[][] course, boolean[] visited, int cur){
    	int res = 0;
    	for(int i = 0;i < course.length;i++){
    		if(!visited[i] && cur + course[i][0] <= course[i][1]){
    			visited[i] = true;
    			res = Math.max(res, 1 + backthrough(course, visited, cur + course[i][0]));
    			visited[i] = false;
    		}
    	}
    	return res;
    }
	
    //Solution2: greedy--AC
    public int scheduleCourse(int[][] courses) {
    	Arrays.sort(courses, (a, b) -> a[1] - b[1]);//greedy, take the courses with early deadlines first
    	PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);
    	int time = 0;
    	for(int[] course : courses){
    		time += course[0];
    		pq.add(course[0]);
    		if(time > course[1])
    			time -= pq.poll();//if time bigger than current deadline, remove the course which costs the longest time
    	}
    	return pq.size();
    }
}
*/
package exercise;
