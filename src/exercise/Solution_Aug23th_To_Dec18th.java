/**
 * @author jhy
 * code from 8.23 to 12.18
 * 33 questions
 */

/**
 * 2017/8/25
 * 377. Combination Sum IV
 * 
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for(int i = 0;i < dp.length;i++){
        	for(int j = 0;j < nums.length;j++){
        		if(i >= nums[j]){
        			dp[i] += dp[i - nums[j]];
        		}
        	}
        }
        return dp[target];
    }
 */

/**
 * 2017/9/6
 * 424. Longest Repeating Character Replacement
 * 
    public int characterReplacement(String s, int k) {
        int start = 0,end = 0,maxCount = 0;
        int result = 0;
        int[] count = new int[26];
        for(end = 0;end < s.length();end++){
        	maxCount = Math.max(maxCount, ++count[s.charAt(end) - 'A']);
        	while(end - start + 1 - maxCount > k){
        		count[s.charAt(start) - 'A']--;
        		start++;
        	}
        	result = Math.max(result, end - start + 1);
        }
        return result;
    }
 */

/**
 * 2017/11/3
 * 390. Elimination Game
 * 
    public int lastRemaining(int n) {
    	if(n <= 0)return n;
        int head = 1,step = 1,remain = n;
        boolean left = true;
        while(remain != 1){
        	if(left || remain % 2 == 1){
        		head += step;
        	}
        	step *= 2;
        	remain /= 2;
        	left = !left;
        }
        return head;
    }
 */

/**
 * 2017/11/27
 * 728. Self Dividing Numbers
 * 
class Solution {
    public List<Integer> selfDividingNumbers(int left, int right) {
        List<Integer> result = new ArrayList<Integer>();
        for(int i = left;i <= right;i++){
        	if(isDividingNumber(i)){
        		result.add(i);
        	}
        }
        return result;
    }
    
    public boolean isDividingNumber(int i){
    	int temp = i;
    	int flag = 0;
    	while(temp > 0){
    		flag = temp % 10;
    		if(temp % 10 == 0)return false;
    		if(i % flag != 0)return false;
    		temp /= 10;
    	}
    	return true;
    }
}

 * 2017/11/27
 * 682. Baseball Game
 * 
class Solution {
    public static boolean isNumeric(String str) {
        // 该正则表达式可以匹配所有的数字 包括负数
        Pattern pattern = Pattern.compile("-?[0-9]+\\.?[0-9]*");
        String bigStr;
        try {
            bigStr = new BigDecimal(str).toString();
        } catch (Exception e) {
            return false;//异常 说明包含非数字。
        }

        Matcher isNum = pattern.matcher(bigStr); // matcher是全匹配
        if (!isNum.matches()) {
            return false;
        }
        return true;
    }

	public int calPoints(String[] ops) {
        int sum = 0;
        Stack<Integer> stack = new Stack<Integer>();
        for(int i = 0;i < ops.length;i++){
        	if(isNumeric(ops[i])){
        		int temp = Integer.parseInt(ops[i]);
        		sum += temp;
        		stack.push(temp);
        	}
        	else if(ops[i].equals("D")){
        		if(!stack.isEmpty()){
        			int temp = stack.peek();
        			temp *= 2;
        			sum += temp;
        			stack.push(temp);
        		}
        	}
        	else if(ops[i].equals("+")){
        		int a = stack.pop();
        		int b = stack.peek();
        		stack.push(a);
        		a += b;
        		sum += a;
        		stack.push(a);
        	}
        	else{
        		int temp = stack.pop();
        		sum -= temp;
        	}
        }
        return sum;
    }
}
 */

/**
 * 2017/11/28
 * 693. Binary Number with Alternating Bits
 * 
	public boolean hasAlternatingBits(int n) {
		if(n % 2 == 1){
			n = n >> 1;
		}
		while(n > 0){
			n -= 2;
			if(n % 4 != 0)return false;
			n = n / 4;
		}
		return true;
	}
	
 * 2017/11/28
 * 690. Employee Importance
 * 
    public int getImportance(List<Employee> employees, int id) {
        int sum = 0;
        for(Employee employee : employees){
        	if(employee.id == id){
        		sum += employee.importance;
        		for(int i:employee.subordinates){
        			sum += getImportance(employees, i);
        		}
        		break;
        	}
        }
        return sum;
    }
    
 * 2017/11/28
 * 695. Max Area of Island
 * 
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int row = grid.length;
        int column = grid[0].length;
        int max = 0;
        for(int i = 0;i < row;i++){
        	for(int j = 0;j < column;j++){
        		if(grid[i][j] == 1){
        			int sum = areaOfIsland(grid, i, j, row, column);
        			max = Math.max(max, sum);
        		}
        	}
        }
        return max;
    }
    
    public int areaOfIsland(int[][] grid,int i,int j,int row,int column){
    	if(i < 0 || i >= row || j < 0 || j >= column || grid[i][j] != 1){
    		return 0;
    	}
    	int sum = 1;
    	grid[i][j] = -1;
    	sum += areaOfIsland(grid, i - 1, j, row, column) + areaOfIsland(grid, i + 1, j, row, column) + areaOfIsland(grid, i, j - 1, row, column) + areaOfIsland(grid, i, j + 1, row, column);
    	return sum;
    }
}

 * 2017/11/28
 * 696. Count Binary Substrings
 * 
    public int countBinarySubstrings(String s) {
        List<Integer> list = new ArrayList<Integer>();
        char index = s.charAt(0);
        int sum = 0;
        for(int i = 0;i < s.length();){
        	sum = 0;
        	while(i < s.length() && s.charAt(i) == index){
        		sum++;
        		i++;
        	}
        	list.add(sum);
        	sum = 0;
        	while(i < s.length() && s.charAt(i) != index){
        		sum++;
        		i++;
        	}
        	list.add(sum);
        }
        int result = 0;
        for(int i = 0;i < list.size() - 1;i++){
        	result += Math.min(list.get(i), list.get(i + 1));
        }
        return result;
    }
 */

/**
 * 2017/11/29
 * 733. Flood Fill
 * 
class Solution {
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        fill(image, sr, sc, image[sr][sc], newColor);
    	return image;
    }
    
    public void fill(int[][] image,int i,int j,int color,int newColor){
    	if(i < 0 || i >= image.length || j < 0 || j >= image[0].length || image[i][j] != color || color == newColor){
    		return;
    	}
    	image[i][j] = newColor;
    	fill(image, i - 1, j, color, newColor);
    	fill(image, i + 1, j, color, newColor);
    	fill(image, i, j - 1, color, newColor);
    	fill(image, i, j + 1, color, newColor);
    }
}
 */

/**
 * 2017/12/1
 * 717. 1-bit and 2-bit Characters
 * 
    public boolean isOneBitCharacter(int[] bits) {
    	if(bits.length <= 1)return true;
    	if(bits[bits.length - 1] == 0){
    		if(bits[bits.length - 2] == 0){
    			return true;    			
    		}
    		else{
    			int num = 0;
    			for(int i = bits.length - 2;i >= 0;i--){
    				if(bits[i] == 1){
    					num++;
    				}
    				else{
    					break;
    				}
    			}
    			if(num % 2 == 0){
    				return true;
    			}
    		}
    	}
    	return false;
    }

 * 2017/12/1
 * 697. Degree of an Array
 * 
    public int findShortestSubArray(int[] nums) {
        Map<Integer, Integer> map = new HashMap<Integer,Integer>();
        int maxDegree = 0;
        for(int num:nums){
        	map.put(num, map.getOrDefault(num, 0) + 1);
        	maxDegree = Math.max(maxDegree, map.get(num));
        }
        List<Integer> list = new ArrayList<Integer>();
        for(int key : map.keySet()){
        	if(map.get(key) == maxDegree){
        		list.add(key);
        	}
        }
        int result = Integer.MAX_VALUE;
        int left = 0;
        int right = 0;
        for(int index : list){
        	for(int i = 0;i < nums.length;i++){
        		if(nums[i] == index){
        			left = i;
        			break;
        		}
        	}
        	for(int i = nums.length - 1;i >= 0;i--){
        		if(nums[i] == index){
        			right = i;
        			break;
        		}
        	}
        	result = Math.min(result, right - left + 1);
        }
        return result;
    }
    
 * 2017/12/1
 * 661. Image Smoother
 * 
class Solution {
    public int[][] imageSmoother(int[][] M) {
        if(M == null)return M;
        if(M[0] == null)return M;
        int row = M.length;
        int column = M[0].length;
        int[][] result = new int[row][column];
        for(int i = 0;i < row;i++){
        	for(int j = 0;j < column;j++){
        		int sum = 0;
        		int num = 0;
        		for(int a : new int[]{-1,0,1}){
        			for(int b : new int[]{-1,0,1}){
        				if(isValid(i+a,j+b,row,column)){
        					sum += M[i + a][j + b];
        					num++;
        				}
        			}
        		}
        		result[i][j] = sum / num;
        	}
        }
        return result;
    }
    
    public boolean isValid(int i,int j,int row,int column){
    	return i >= 0 && j >= 0 && i < row && j < column;
    }
}
 */

/**
 * 2017/12/5
 * 669. Trim a Binary Search Tree
 * 
    public TreeNode trimBST(TreeNode root, int L, int R) {
        while(root != null && (root.val < L || root.val > R)){
        	if(root.val < L)root = root.right;
        	else root = root.left;
        }
        if(root == null)return root;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        while(!queue.isEmpty()){
        	int num = queue.size();
        	for(int i = 0;i < num;i++){
        		TreeNode temp = queue.poll();
        		while(temp.left != null && temp.left.val < L){
        			temp.left = temp.left.right;
        		}
        		while(temp.right != null && temp.right.val > R){
        			temp.right = temp.right.left;
        		}
        		if(temp.left != null)queue.add(temp.left);
        		if(temp.right != null)queue.add(temp.right);
        	}
        }
        return root;
    }
 */

/**
 * 2017/12/6
 * 674. Longest Continuous Increasing Subsequence
 * 
    public int findLengthOfLCIS(int[] nums) {
    	if(nums.length == 0)return 0;
        int num = 1;
        int result = 0;
        for(int i = 1;i < nums.length;i++){
        	if(nums[i] > nums[i - 1]){
        		num++;
        	}
        	else{
        		result = Math.max(num, result);
        		num = 1;
        	}
        }
        result = Math.max(num, result);
        return result;
    }
    
 * 2017/12/6
 * 671. Second Minimum Node In a Binary Tree
 * 
    public int findSecondMinimumValue(TreeNode root) {
    	if(root == null)return -1;
    	if(root.left == null && root.right == null)return -1;
    	int left = root.left.val;
    	int right = root.right.val;
    	if(left == root.val){
    		left = findSecondMinimumValue(root.left);
    	}
    	if(right == root.val){
    		right = findSecondMinimumValue(root.right);
    	}
    	if(left == -1 && right == -1)return -1;
    	else if(left == -1)return right;
    	else if(right == -1)return left;
    	else{
    		return Math.min(left, right);
    	}
    }
    
 * 2017/12/6
 * 720. Longest Word in Dictionary
 * 
    public String longestWord(String[] words) {
        Arrays.sort(words);
        String res = "";
        Set<String> set = new HashSet<String>();
        for(String word:words){
        	if(word.length() == 1 || set.contains(word.substring(0,word.length() - 1))){
                res = word.length() > res.length() ? word : res;
                set.add(word);
        	}
        }
        return res;
    }
    
 * 2017/12/6
 * 724. Find Pivot Index
 * 
    public int pivotIndex(int[] nums) {
    	if(nums.length == 0)return -1;
    	if(nums.length == 1)return -1;
    	int left = 0;
    	int right = 0;
    	for(int i = 1;i < nums.length;i++){
    		right += nums[i];
    	}
    	if(left == right)return 0;
    	for(int i = 0;i < nums.length - 1;i++){
    		left += nums[i];
    		right -= nums[i + 1];
    		if(left == right)return i + 1;
    	}
    	return -1;
    }
 */

/**
 * 2017/12/7
 * 734. Sentence Similarity
 * 
    public boolean areSentencesSimilar(String[] words1, String[] words2, String[][] pairs) {
        if(words1.length != words2.length)return false;
        Map<String, List<String>> map = new HashMap<>();
        for(int i = 0;i < pairs.length;i++){
        	if(!map.containsKey(pairs[i][0])){
        		List<String> list = new ArrayList<>();
        		list.add(pairs[i][1]);
        		map.put(pairs[i][0], list);
        	}
        	else{
        		List<String> list = map.get(pairs[i][0]);
        		list.add(pairs[i][1]);
        		map.put(pairs[i][0], list);
        	}

        }
        for(int i = 0;i < words1.length;i++){
        	if(!words1[i].equals(words2[i])){
        		if(map.containsKey(words1[i]) && map.get(words1[i]).contains(words2[i]))
        			continue;
        		else if(map.containsKey(words2[i]) && map.get(words2[i]).contains(words1[i]))
        			continue;
        		return false;
        	}
        }
        return true;
    }
    
 * 2017/12/7
 * 443. String Compression
 * 
class Solution {
    public int compress(char[] chars) {
        if(chars.length <= 1)return chars.length;
        List<Character> list = new ArrayList<>();
        int num = 1;
        char temp = chars[0];
        for(int i = 1;i < chars.length;i++){
        	if(chars[i] == temp){
        		num++;
        	}
        	else{
        		if(num == 1){
        			temp = chars[i];
        			list.add(chars[i - 1]);
        		}
        		else{
        			temp = chars[i];
        			list.add(chars[i - 1]);
        			pushBit(num, list);
        			num = 1;
        		}
        	}
        }
		if(num == 1){
			list.add(chars[chars.length - 1]);
		}
		else{
			list.add(chars[chars.length - 1]);
			pushBit(num, list);
		}
        for(int i = 0;i < list.size();i++){
        	chars[i] = list.get(i);
        }
        return list.size();
    }
    
    public void pushBit(int bit,List<Character> list){
    	Stack<Character> stack = new Stack<>();
    	while(bit > 0){
    		stack.push((char)(bit % 10 + '0'));
    		bit = bit / 10;
    	}
    	while(!stack.isEmpty()){
    		list.add(stack.pop());
    	}
    }
}

 * 2017/12/7
 * 686. Repeated String Match
 * 
class Solution {
    public int repeatedStringMatch(String A, String B) {
        int[] next = new int[B.length()];
        nextKMP(B, next);
    	String temp = new String(A);
    	int res = 1;
    	while(temp.length() < B.length()){
    		temp += A;
    		res++;
    	}
    	if(isSubstringKMP(temp, B, next))return res;
		while (res < 2 || temp.length() <= 2 * B.length()) {
			temp += A;
			res++;
			if (isSubstringKMP(temp, B, next))
				return res;
		}
    	return -1;
    }
    
    public void nextKMP(String p,int[] next){//判断p是否为s子串
    	int j = 0;
    	int k = -1;
    	next[0] = -1;
    	while(j < p.length() - 1){
    		if(k == -1 || p.charAt(j) == p.charAt(k)){
    			j++;
    			k++;
    			if(p.charAt(j) != p.charAt(k)){
    				next[j] = k;
    			}
    			else{
    				next[j] = next[k];
    			}
    		}
    		else{
    			k = next[k];
    		}
    	}
    }
    
	public boolean isSubstringKMP(String S, String P, int[] next) {//KMP算法判断P是否为S子串
		int i = 0;
		int j = 0;
		while(i < S.length() && j < P.length()){
			if(j == -1 || S.charAt(i) == P.charAt(j)){
				i++;
				j++;
			}
			else{
				j = next[j];
			}
		}
		if(j == P.length())return true;
		return false;
	}
}
 */

/**
 * 2017/12/8
 * 687. Longest Univalue Path
 * 
class Solution {
	public int res;
    public int longestUnivaluePath(TreeNode root) {
    	res = 0;
        recursiveLongestPath(root);
    	return res;
    }
    
    public int recursiveLongestPath(TreeNode root){
    	if(root == null)return 0;
    	int left = recursiveLongestPath(root.left);
    	int right = recursiveLongestPath(root.right);
    	int leftNum = 0;
    	int rightNum = 0;
    	if(root.left != null && root.val == root.left.val)leftNum = left + 1;
    	if(root.right != null && root.val == root.right.val)rightNum = right + 1;
    	res = Math.max(res, rightNum + leftNum);
    	return Math.max(leftNum, rightNum);
    }
}
 */

/**
 * 2017/12/9
 * 680. Valid Palindrome II
 * 
	public boolean validPalindrome(String s) {
		//aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga
		int i = 0;
		int j = s.length() - 1;
		while (i < j) {
			if (s.charAt(i) == s.charAt(j)) {
				i++;
				j--;
			} 
			else {
				if (i + 1 == j)
					return true;
				int left = i;
				int right = j;
				boolean flag1 = false;
				boolean flag2 = false;
				//若从左或者右边删掉字符均可运行，则需取或
				if (s.charAt(i + 1) == s.charAt(j)) {
					i++;
					while (i < j) {
						if (s.charAt(i) != s.charAt(j))
							break;
						i++;
						j--;
					}
					if(i >= j){
						flag1 = true;
					}
				}
				i = left;
				j = right;
				if (s.charAt(i) == s.charAt(j - 1)) {
					j--;
					while (i < j) {
						if (s.charAt(i) != s.charAt(j))
							break;
						i++;
						j--;
					}
					if(i >= j){
						flag2 = true;
					}
				}
				return flag1 || flag2;
			}
		}
		return true;
	}
	
 * 2017/12/9
 * 665. Non-decreasing Array
 * 
	public boolean checkPossibility(int[] nums) {
		if (nums.length < 3)
			return true;
		boolean flag = true;
		for (int i = 1; i < nums.length; i++) {
			if (nums[i] < nums[i - 1]) {
				if (flag) {
					flag = false;
					if (i == 1) {
						nums[0] = nums[1];
					} else if (nums[i] < nums[i - 2]) {
						nums[i] = nums[i - 1];
					} else {
						nums[i - 1] = nums[i - 2];
					}
				} else {
					return false;
				}
			}
		}
		return true;
	}
 */

/**
 * 2017/12/14
 * 739. Daily Temperatures
 * 
    public int[] dailyTemperatures(int[] temperatures) {
        int[] res = new int[temperatures.length];
        for(int i = 0;i < temperatures.length;i++){
        	for(int j = i + 1;j < temperatures.length;j++){
        		if(temperatures[j] > temperatures[i]){
        			res[i] = j - i;
        			break;
        		}
        	}
        }
        return res;
    }
 */

/**
 * 2017/12/15
 * 667. Beautiful Arrangement II
 * 
    public int[] constructArray(int n, int k) {
        int[] res = new int[n];
        int i = 0;
        for(i = 0;i < n - k - 1;i++){
        	res[i] = i + 1;
        }
    	int left = n - k;
        int right = n;
        while(true){
        	if(i < res.length){
        		res[i++] = left++;
        	}
        	else{
        		break;
        	}
        	if(i < res.length){
        		res[i++] = right--;
        	}
        	else{
        		break;
        	}
        }
        return res;
    }
    
 * 2017/12/15
 * 712. Minimum ASCII Delete Sum for Two Strings
 * 
    public int minimumDeleteSum(String s1, String s2) {
        int[][] dp = new int[s1.length() + 1][s2.length() + 1];
        for(int i = s1.length() - 1;i >= 0;i--){
        	dp[i][s2.length()] = dp[i + 1][s2.length()] + s1.codePointAt(i);
        }
        for(int j = s2.length() - 1;j >= 0;j--){
        	dp[s1.length()][j] = dp[s1.length()][j + 1] + s2.codePointAt(j);
        }
        for(int i = s1.length() - 1;i >= 0;i--){
        	for(int j = s2.length() - 1;j >= 0;j--){
        		if(s1.codePointAt(i) == s2.codePointAt(j)){
        			dp[i][j] = dp[i + 1][j + 1];
        		}
        		else{
        			dp[i][j] = Math.min(dp[i + 1][j] + s1.codePointAt(i), dp[i][j + 1] + s2.codePointAt(j));
        		}
        	}
        }
        return dp[0][0];
    }
    
 * 2017/12/15
 * 725. Split Linked List in Parts
 * 
    public ListNode[] splitListToParts(ListNode root, int k) {
        ListNode[] res = new ListNode[k];
        int sum = 0;
        ListNode temp = root;
        while(temp != null){
        	sum++;
        	temp = temp.next;
        }
        if(sum <= k){
        	for(int i = 0;i < sum;i++){
        		res[i] = root;
        		root = root.next;
        		res[i].next = null;
        	}
        	return res;
        }
        int avg = sum / k;
        int left = avg + 1;
        int right = avg;
        int leftNum = sum % k;
        int rightNum = k - leftNum;
        int index = 0;
        ListNode pre = root;
        temp = root;
        for(int i = 0;i < leftNum;i++){
        	res[index++] = temp;
        	for(int j = 0;j < left;j++){
        		pre = temp;
        		temp = pre.next;
        	}
        	pre.next = null;
        }
        for(int i = 0;i < rightNum;i++){
        	res[index++] = temp;
        	for(int j = 0;j < right;j++){
        		pre = temp;
        		temp = pre.next;
        	}
        	pre.next = null;
        }
        return res;
    }
 */

/**
 * 2017/12/16
 * 672. Bulb Switcher II
 * 
    public int flipLights(int n, int m) {
    	n = Math.min(n, 3);
    	if(m == 0)return 1;
    	if(m == 1)return n == 1 ? 2 : n == 2 ? 3 : 4;
    	if(m == 2)return n == 1 ? 2 : n == 2 ? 4 : 7;
    	return n == 1 ? 2 : n == 2 ? 4 : 8;
    }
 */

/**
 * 2017/12/17
 * 636. Exclusive Time of Functions
 * 
    public int[] exclusiveTime(int n, List<String> logs) {
        int[] res = new int[n];
        Stack<Integer> stack = new Stack<>();
        String[] str = logs.get(0).split(":");
        stack.add(Integer.parseInt(str[0]));
        int pre = Integer.parseInt(str[2]);
        for(int i = 1;i < logs.size();i++){
        	str = logs.get(i).split(":");
        	if(str[1].equals("start")){
        		if(!stack.isEmpty())
        			res[stack.peek()] += Integer.parseInt(str[2]) - pre;
        		stack.push(Integer.parseInt(str[0]));
        		pre = Integer.parseInt(str[2]);
        	}
        	else{
        		res[stack.pop()] += Integer.parseInt(str[2]) - pre + 1;
        		pre = Integer.parseInt(str[2]) + 1;
        	}
        }
        return res;
    }
    
 * 2017/12/17
 * 78. Subsets
 * 
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        res.add(new ArrayList<Integer>());
        for(int i = 0;i < nums.length;i++){
        	List<List<Integer>> pre = new ArrayList<>();
        	for(List<Integer> list : res){
        		pre.add(new ArrayList<>(list));
        		list.add(nums[i]);
        		pre.add(list);
        	}
        	res = pre;
        }
        return res;
    }
    
 * 2017/12/17
 * 621. Task Scheduler
 * 
    public int leastInterval(char[] tasks, int n) {
        int res = 0;
        int[] letterNum = new int[26];
        for(int i = 0;i < tasks.length;i++){
        	letterNum[tasks[i] - 'a']++;
        }
        Arrays.sort(letterNum);
        while(letterNum[25] > 0){
        	int i = 0;
        	while(i <= n){
        		if(letterNum[25] == 0)break;
        		if(i < 26 && letterNum[25 - i] > 0)letterNum[25 - i]--;
        		i++;
        		res++;
        	}
        	Arrays.sort(letterNum);
        }
        return res;
    }
 */

/**
 * 2017/12/18
 * 750. Number Of Corner Rectangles
 * 
    public int countCornerRectangles(int[][] grid) {
        int res = 0;
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int[] row : grid){
        	for(int i = 0;i < row.length;i++){
        		if(row[i] == 1){
        			for(int j = i + 1;j < row.length;j++){
        				if(row[j] == 1){
        					int count = i * 200 + j;
        					int c = map.getOrDefault(count, 0);
        					res += c;
        					map.put(count, c + 1);
        				}
        			}
        		}
        	}
        }
        return res;
    }
    
 * 2017/12/18
 * 638. Shopping Offers
 * 
    public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
        int result = Integer.MAX_VALUE;
        for(int i = 0;i < special.size();i++){
        	int size = Integer.MAX_VALUE;
        	List<Integer> offer = special.get(i);
        	boolean isPositive = true;
        	for(int j = 0;j < needs.size();j++){
        		int temp = needs.get(j) - offer.get(j);
        		if(temp < 0 && isPositive)isPositive = false;
        		if(offer.get(j) > 0)
        			size = Math.min(size, needs.get(j) / offer.get(j));
        	}
        	for(int j = 0;j < needs.size();j++){
        		needs.set(j, needs.get(j) - size * offer.get(j));
        	}
        	if(isPositive)
        		result = Math.min(result, shoppingOffers(price, special, needs) + size * offer.get(needs.size()));
        	for(int j = 0;j < needs.size();j++){
        		int temp = needs.get(j) + offer.get(j) * size;
        		needs.set(j, temp);
        	}
        }
        int retail = 0;
        for(int i = 0;i < needs.size();i++){
        	retail += price.get(i) * needs.get(i);
        }
        return Math.min(retail, result);
    }
 */
package exercise;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.function.Predicate;


class Solution_Aug23th_To_Dec18th {

}







