/**
 * @author jhy code from 1.20 to 2.3
 * 34 questions
 */

/**
 * 2018/1/20
 * 103. Binary Tree Zigzag Level Order Traversal
 * 
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        Stack<TreeNode> left = new Stack<>();
        Stack<TreeNode> right = new Stack<>();
        List<List<Integer>> res = new ArrayList<>();
        if(root == null)return res;
        List<Integer> first = new ArrayList<>();
        first.add(root.val);
        res.add(first);
        right.add(root);
        while(!right.isEmpty()){
        	List<Integer> temp = new ArrayList<>();
        	while(!right.isEmpty()){
        		TreeNode node = right.pop();
        		if(node.right != null){
        			temp.add(node.right.val);
        			left.add(node.right);
        		}
        		if(node.left != null){
        			temp.add(node.left.val);
        			left.add(node.left);
        		}
        	}
        	if(!temp.isEmpty())res.add(temp);
        	temp = new ArrayList<>();
        	while(!left.isEmpty()){
        		TreeNode node = left.pop();
        		if(node.left != null){
        			temp.add(node.left.val);
        			right.add(node.left);
        		}
        		if(node.right != null){
        			temp.add(node.right.val);
        			right.add(node.right);
        		}
        	}
        	if(!temp.isEmpty())res.add(temp);
        }
        return res;
    }
 */

/**
 * 2018/1/21
 * 659. Split Array into Consecutive Subsequences
 * 
    public boolean isPossible(int[] nums) {
    	Map<Integer, Integer> fre = new HashMap<>();
    	Map<Integer, Integer> exist = new HashMap<>();
    	for(int num : nums){
    		fre.put(num, fre.getOrDefault(num, 0) + 1);
    	}
    	for(int num : nums){
    		if(fre.get(num) == 0)continue;
    		else if(exist.getOrDefault(num, 0) > 0){
    			exist.put(num, exist.get(num) - 1);
    			exist.put(num + 1, exist.getOrDefault(num + 1, 0) + 1);
    		}
    		else if(fre.getOrDefault(num + 1, 0) > 0 && fre.getOrDefault(num + 2, 0) > 0){
    			fre.put(num + 1, fre.get(num + 1) - 1);
    			fre.put(num + 2, fre.get(num + 2) - 1);
    			exist.put(num + 3, exist.getOrDefault(num + 3, 0) + 1);
    		}
    		else return false;
    		fre.put(num, fre.get(num) - 1);
    	}
    	return true;
    }
    
 * 2018/1/21
 * 114. Flatten Binary Tree to Linked List
 * 
class Solution {
	TreeNode cur;
    public void flatten(TreeNode root) {
    	if(root == null)return;
        cur = root;
        TreeNode left = root.left;
        TreeNode right = root.right;
        preOrder(left);
        preOrder(right);
    }
    
    public void preOrder(TreeNode root){
    	if(root == null)return;
    	TreeNode left = root.left;
    	TreeNode right = root.right;
    	cur.right = root;
    	cur.left = null;
    	cur = cur.right;
    	preOrder(left);
    	preOrder(right);
    }
}

 * 2018/1/21
 * 649. Dota2 Senate
 * 
    public String predictPartyVictory(String senate) {
        Queue<Integer> R = new LinkedList<>();
        Queue<Integer> D = new LinkedList<>();
        int len = senate.length();
        for(int i = 0;i < len;i++){
        	if(senate.charAt(i) == 'R'){
        		R.add(i);
        	}
        	else{
        		D.add(i);
        	}
        }
        while(!R.isEmpty() && !D.isEmpty()){
        	int r = R.poll();
        	int d = D.poll();
        	if(r < d){
        		R.add(r + len);
        	}
        	else{
        		D.add(d + len);
        	}
        }
        return R.size() > D.size() ? "Radiant" : "Dire";
    }
 */

/**
 * 2018/1/22
 * 200. Number of Islands
 * 
class Solution {
    public int numIslands(char[][] grid) {
        int count = 0;
        if(grid.length == 0 || grid[0].length == 0)return count;
        for(int i = 0;i < grid.length;i++){
        	for(int j = 0;j < grid[i].length;j++){
        		if(grid[i][j] == '1'){
        			deepSearch(i, j, grid);
        			count++;
        		}
        	}
        }
        return count;
    }
    
    public void deepSearch(int i, int j,char[][] grid){//将属于一块land的元素置一
    	if(i < 0 || j < 0 || i >= grid.length || j >= grid[i].length || grid[i][j] == '0')return;
    	grid[i][j] = '0';
    	deepSearch(i - 1, j, grid);
    	deepSearch(i + 1, j, grid);
    	deepSearch(i, j - 1, grid);
    	deepSearch(i, j + 1, grid);
    }
}

 * 2018/1/22
 * 375. Guess Number Higher or Lower II
 * 
class Solution {
    public int getMoneyAmount(int n) {
    	int[][] pay = new int[n + 1][n + 1];
    	return minPay(pay, 1, n);
    }
    
    public int minPay(int[][] pay, int start, int end){
    	if(start >= end)return 0;
    	if(pay[start][end] != 0)return pay[start][end];
    	int res = Integer.MAX_VALUE;
    	for(int i = start;i <= end;i++){
    		int temp = i + Math.max(minPay(pay, start, i - 1), minPay(pay, i + 1, end));
    		res = Math.min(temp, res);
    	}
    	pay[start][end] = res;
    	return res;
    }
}

 * 2018/1/22
 * 766. Toeplitz Matrix
 * 
class Solution {
    public boolean isToeplitzMatrix(int[][] matrix) {
        if(matrix.length == 0 || matrix[0].length == 0)return true;
        for(int i = matrix.length - 1;i >= 0;i--){
        	if(!diagonalIsEqual(matrix, i, 0))return false;
        }
    	for(int j = 1;j < matrix[0].length;j++){
    		if(!diagonalIsEqual(matrix, 0, j))return false;
    	}
    	return true;
    }
    
    public boolean diagonalIsEqual(int[][] matrix, int i, int j) {
		int temp = matrix[i][j];
		while(i < matrix.length && j < matrix[i].length){
			if(matrix[i++][j++] != temp)return false;
		}
    	return true;
	}
}

 * 2018/1/22
 * 17. Letter Combinations of a Phone Number
 * 
	public List<String> letterCombinations(String digits) {
		LinkedList<String> ans = new LinkedList<String>();
		if (digits.isEmpty())
			return ans;
		String[] mapping = new String[] { "0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
		ans.add("");
		for (int i = 0; i < digits.length(); i++) {
			int x = Character.getNumericValue(digits.charAt(i));
			while (ans.peek().length() == i) {
				String t = ans.remove();
				for (char s : mapping[x].toCharArray())
					ans.add(t + s);
			}
		}
		return ans;
	}
	
 * 2018/1/22
 * 395. Longest Substring with At Least K Repeating Characters
 * 
class Solution {
    public int longestSubstring(String s, int k) {
        char[] str = s.toCharArray();
        return helper(str, 0, s.length(),k);
    }
    
    public int helper(char[] str,int start,int end,int k){
    	if(start >= end)return 0;
    	int[] count = new int [26];
    	for(int i = start;i < end;i++){
    		count[str[i] - 'a']++;
    	}
    	for(int i = 0;i < 26;i++){
    		if(count[i] < k && count[i] > 0){
    			for(int j = start;j < end;j++){
    				if(str[j] - 'a' ==  + i){
    					return Math.max(helper(str, start, j, k), helper(str, j + 1, end, k));
    				}
    			}
    		}
    	}
    	return end - start;
    }
}
 */

/**
 * 2018/1/23
 * 376. Wiggle Subsequence
 * 
	public int wiggleMaxLength(int[] nums) {
		if (nums.length == 0 || nums.length == 1) {
			return nums.length;
		}
		int k = 0;
		while (k < nums.length - 1 && nums[k] == nums[k + 1]) { 
			k++;
		}
		if (k == nums.length - 1) {
			return 1;
		}
		int result = 2; // This will track the result of result array
		boolean smallReq = nums[k] < nums[k + 1]; // To check series starting
													// pattern
		for (int i = k + 1; i < nums.length - 1; i++) {
			if (smallReq && nums[i + 1] < nums[i]) {
				nums[result] = nums[i + 1];
				result++;
				smallReq = !smallReq; // Toggle the requirement from small to
										// big number
			} else {
				if (!smallReq && nums[i + 1] > nums[i]) {
					nums[result] = nums[i + 1];
					result++;
					smallReq = !smallReq; // Toggle the requirement from big to
											// small number
				}
			}
		}
		return result;
	}
 */

/**
 * 2018/1/24
 * 769. Max Chunks To Make Sorted
 * 
    public int maxChunksToSorted(int[] arr) {
        int res = 0;
        Set<Integer> set = new HashSet<>();
        for(int i = 0;i < arr.length;i++){
        	if(set.contains(i)){
        		set.remove(i);
        	}
        	else{
        		set.add(i);
        	}
        	if(set.contains(arr[i])){
        		set.remove(arr[i]);
        	}
        	else{
        		set.add(arr[i]);
        	}
        	if(set.isEmpty())res++;
        }
        return res;
    }
    
 * 2018/1/24
 * 299. Bulls and Cows
 * 
    public String getHint(String secret, String guess) {
        int a = 0;
        int b = 0;
        Map<Character, Integer> map = new HashMap<>();
        for(int i = 0;i < secret.length();i++){
        	if(secret.charAt(i) == guess.charAt(i)){
        		a++;
        	}
        	else{
        		map.put(secret.charAt(i), map.getOrDefault(secret.charAt(i), 0) + 1);
        	}
        }
        for(int i = 0;i < secret.length();i++){
        	if(secret.charAt(i) != guess.charAt(i)){
        		if(map.containsKey(guess.charAt(i)) && map.get(guess.charAt(i)) > 0){
        			map.put(guess.charAt(i), map.get(guess.charAt(i)) - 1);
        			b++;
        		}
        	}
        }
        return a + "A" + b + "B";
    }
 */

/**
 * 2018/1/25
 * 764. Largest Plus Sign
 * 
    public int orderOfLargestPlusSign(int N, int[][] mines) {
        int[][] arr = new int[N][N];
        for(int i = 0;i < N;i++){
        	Arrays.fill(arr[i],N);
        }
        for(int[] mine : mines){
        	arr[mine[0]][mine[1]] = 0;
        }
        for(int i = 0;i < N;i++){
        	for(int j = 0,k = N - 1,l = 0,r = 0,u = 0,d = 0; j < N; j++,k--){
        		arr[i][j] = Math.min(arr[i][j], l = (arr[i][j] == 0 ? 0 : l + 1));
        		arr[i][k] = Math.min(arr[i][k], r = (arr[i][k] == 0 ? 0 : r + 1));
        		arr[j][i] = Math.min(arr[j][i], u = (arr[j][i] == 0 ? 0 : u + 1));
        		arr[k][i] = Math.min(arr[k][i], d = (arr[k][i] == 0 ? 0 : d + 1));
        	}
        }
        int res = 0;
        for(int i = 0;i < N;i++){
        	for(int j = 0;j < N;j++){
        		res = Math.max(res, arr[i][j]);
        	}
        }
        return res;
    }
    
 * 2018/1/25
 * 658. Find K Closest Elements
 * 
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
    	List<Integer> res = new ArrayList<>();
    	int low = 0;
    	int high = arr.length - k;
    	while(low < high){
    		int mid = (low + high) / 2;
    		if(x - arr[mid] > arr[mid + k] - x){
    			low = mid + 1;
    		}
    		else{
    			high = mid;
    		}
    	}
    	for(int i = low;i < low + k;i++){
    		res.add(arr[i]);
    	}
    	return res;
    }
    
 * 2018/1/25
 * 473. Matchsticks to Square
 * 
public class Solution {
    public boolean makesquare(int[] nums) {
    	if (nums == null || nums.length < 4) return false;
        int sum = 0;
        for (int num : nums) sum += num;
        if (sum % 4 != 0) return false;
        
    	return dfs(nums, new int[4], 0, sum / 4);
    }
    
    private boolean dfs(int[] nums, int[] sums, int index, int target) {
    	if (index == nums.length) {
    	    if (sums[0] == target && sums[1] == target && sums[2] == target) {
    		return true;
    	    }
    	    return false;
    	}
    	
    	for (int i = 0; i < 4; i++) {
    	    if (sums[i] + nums[index] > target) continue;
    	    sums[i] += nums[index];
            if (dfs(nums, sums, index + 1, target)) return true;
    	    sums[i] -= nums[index];
    	}
    	
    	return false;
    }
}
 */

/**
 * 2018/1/26
 * 40. Combination Sum II
 * 
class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if(candidates.length == 0)return res;
        Arrays.sort(candidates);
        reverse(candidates);
        help(candidates, target, 0, new ArrayList<>(), res);
        return res;
    }
    
    public void help(int[] candidates,int target,int start, List<Integer> sum,List<List<Integer>> res){//help函数计算res
    	if(target == 0){
    		List<Integer> temp = new ArrayList<>(sum);
    		if(!isDuplicate(res, temp)){
    			res.add(temp);
    		}
    		return;
    	}
    	for(int i = start;i < candidates.length;i++){
    		sum.add(candidates[i]);
    		if(target >= candidates[i])
    			help(candidates, target - candidates[i], i + 1, sum, res);
    		sum.remove(sum.size() - 1);
    	}
    }
    
    public boolean isDuplicate(List<List<Integer>> res,List<Integer> temp){//判断temp是否与res中结果重复
    	for(List<Integer> re : res){
    		if(re.size() == temp.size()){
    			boolean flag = true;
    			for(int i = 0;i < re.size();i++){
    				if(re.get(i) != temp.get(i)){
    					flag = false;
    					break;
    				}
    			}
    			if(flag)return true;
    		}
    	}
    	return false;
    }
    
    public void reverse(int[] nums){//对数组进行翻转
    	int temp;
    	for(int i = 0, j = nums.length - 1;i < nums.length / 2;i++,j--){
    		temp = nums[i];
    		nums[i] = nums[j];
    		nums[j] = temp;
    	}
    }
}

 * 2018/1/26
 * 109. Convert Sorted List to Binary Search Tree
 * 
class Solution {
    public TreeNode sortedListToBST(ListNode head) {
        List<Integer> list = new ArrayList<>();
        while(head != null){
        	list.add(head.val);
        	head = head.next;
        }
        return dfs(list, 0, list.size());
    }
    
    public TreeNode dfs(List<Integer> list,int start,int end){
    	if(start >= end)return null;
    	int mid = (start + end) / 2;
    	TreeNode root = new TreeNode(list.get(mid));
    	root.left = dfs(list, start, mid);
    	root.right = dfs(list, mid + 1, end);
    	return root;
    }
}

 * 2018/1/26
 * 274. H-Index
 * 
    // O(N) time; O(N) space
    public int hIndex(int[] citations) {
        int[] papers = new int[citations.length + 1];
        for (int c : citations) {
            papers[Math.min(c, citations.length)]++; // attention
        }
        int h = citations.length;
        int nPaper = papers[h];
        while (nPaper < h) {
            h--;
            nPaper += papers[h];
        }
        return h;
    }

 * 2018/1/26
 * 275. H-Index II
 * 
    public int hIndex(int[] citations) {
        int left = 0;
        int right = citations.length - 1;
        while(left <= right){
        	int mid = (left + right) / 2;
        	if(citations.length - mid <= citations[mid]){
        		right = mid - 1;
        	}
        	else{
        		left = mid + 1;
        	}
        }
        return citations.length - left;
    }
    
 * 2018/1/26
 * 74. Search a 2D Matrix
 * 
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        if(matrix.length == 0 || matrix[0].length == 0)return false;
        int len = matrix[0].length;
        int left = code(len, 0, 0);
        int right = code(len, matrix.length - 1, len - 1);
        while(left <= right){
        	int mid = (left + right) / 2;
        	int[] temp = decode(mid, len);
        	if(matrix[temp[0]][temp[1]] > target){
        		right = mid - 1;
        	}
        	else if(matrix[temp[0]][temp[1]] < target){
        		left = mid + 1;
        	}
        	else{
        		return true;
        	}
        }
        return false;
    }
    
    public int[] decode(int code,int len){
    	int[] res = new int[2];
    	res[0] = code / len;//row
    	res[1] = code % len;//colunm
    	return res;
    }
    
    public int code(int len,int row,int colunm){
    	return row * len + colunm;
    }
}

 * 2018/1/26
 * 767. Reorganize String
 * 
class TestCompare implements Comparator<int[]> {
    public static Comparator<int[]> testcompare = new TestCompare();
	public int compare(int[] a,int[] b){
		if(a[1] < b[1])return -1;
		else if(a[1] > b[1])return 1;
		else{
			return a[2] - b[2];
		}
	}
}

class Solution {
    public String reorganizeString(String S) {
        int[][] count = new int[26][3];//[0]代表哪个字母,[1]代表出现次数,[2]代表顺序(使快排稳定)
        for(int i = 0;i < 26;i++){
        	count[i][0] = i;
        }
        for(int i = 0;i < S.length();i++){
        	count[S.charAt(i) - 'a'][1]++;
        }
        Arrays.sort(count, TestCompare.testcompare);
        for(int i = 0;i < 26;i++){
        	count[i][2] = i;
        }
        if(count[25][1] > (S.length() + 1) / 2)return "";
        int num = 0;
        StringBuffer str = new StringBuffer();
        while(num < S.length() - 1){
        	num += 2;
        	str.append((char)(count[25][0] + 'a')).append((char)(count[24][0] + 'a'));
        	count[25][1]--;
        	count[24][1]--;
            Arrays.sort(count, TestCompare.testcompare);
            for(int i = 0;i < 26;i++){
            	count[i][2] = i;
            }
        }
        if(num < S.length()){
        	str.append((char)(count[25][0] + 'a'));
        }
        return str.toString();
    }
}
 */

/**
 * 2018/1/27
 * 50. Pow(x, n)
 * 
    public double myPow(double x, int n) {
        double ans = 1;
        long absN = Math.abs((long)n);
        while(absN > 0) {
            if((absN&1)==1) ans *= x;
            absN >>= 1;
            x *= x;
        }
        return n < 0 ?  1/ans : ans;
    }
 */

/**
 * 2018/1/28
 * 120. Triangle
 * 
    public int minimumTotal(List<List<Integer>> triangle) {
    	int len = triangle.size();
    	if(len == 0)return 0;
    	if(len == 1)return triangle.get(0).get(0);
    	int[] cur = new int[0];
    	int[] pre = new int[1];
    	pre[0] = triangle.get(0).get(0);
    	for(int i = 1;i < len;i++){
    		cur = new int[i + 1];
    		List<Integer> temp = triangle.get(i);
    		for(int j = 1;j < temp.size() - 1;j++){
    			cur[j] = Math.min(pre[j - 1], pre[j]) + temp.get(j);
    		}
    		cur[0] = pre[0] + temp.get(0);
    		cur[i] = pre[i - 1] + temp.get(i);
    		pre = cur;
    	}
    	int res = Integer.MAX_VALUE;
    	for(int i = 0;i < cur.length;i++){
    		res = Math.min(res, cur[i]);
    	}
    	return res;
    }
    
 * 2018/1/28
 * 372. Super Pow
 * 
class Solution {
	public final int mod = 1337;
    public int superPow(int a, int[] b) {
        return help(a, b, b.length);
    }
    
    public int help(int a,int[] b,int end){
    	if(end == 0)return 1;
    	int num = b[end - 1];
    	return (power(help(a, b, end - 1), 10) * power(a, num)) % mod;
    }
    
    public int power(int a, int b){
    	a %= mod;
    	int res = 1;
    	while(b > 0){
    		res = (res * a) % mod;
    		b--;
    	}
    	return res;
    }
}

 * 2018/1/28
 * 213. House Robber II
 * 
class Solution {
    public int rob(int[] nums) {
        if(nums.length == 0)return 0;
        return Math.max(help(nums, 0, nums.length - 1), help(nums, 1, nums.length)); 
    }
    
    public int help(int[] nums,int start,int end){
    	int rob = nums[start];
    	int notRob = 0;
    	for(int i = start + 1;i < end;i++){
    		int cur = notRob;
    		notRob = rob;
    		rob = Math.max(rob, cur + nums[i]);
    	}
    	return Math.max(rob, notRob);
    }
}
 */

/**
 * 2018/1/29
 * 771. Jewels and Stones
 * 
    public int numJewelsInStones(String J, String S) {
        Set<Character> set = new HashSet<>();
        for(int i = 0;i < J.length();i++){
        	set.add(J.charAt(i));
        }
        int res = 0;
        for(int i = 0;i < S.length();i++){
        	if(set.contains(S.charAt(i)))res++;
        }
        return res;
    }

 * 2018/1/29
 * 393. UTF-8 Validation
 * 
	public boolean validUtf8(int[] data) {
		if(data==null || data.length==0) return false;
		boolean isValid = true;
		for(int i=0;i<data.length;i++) {
			if(data[i]>255) return false; // 1 after 8th digit, 100000000
			int numberOfBytes = 0;
			if((data[i] & 128) == 0) { // 0xxxxxxx, 1 byte, 128(10000000)
				numberOfBytes = 1;
			} else if((data[i] & 224) == 192) { // 110xxxxx, 2 bytes, 224(11100000), 192(11000000)
				numberOfBytes = 2;
			} else if((data[i] & 240) == 224) { // 1110xxxx, 3 bytes, 240(11110000), 224(11100000)
				numberOfBytes = 3;
			} else if((data[i] & 248) == 240) { // 11110xxx, 4 bytes, 248(11111000), 240(11110000)
				numberOfBytes = 4;
			} else {
				return false;
			}
			for(int j=1;j<numberOfBytes;j++) { // check that the next n bytes start with 10xxxxxx
				if(i+j>=data.length) return false;
				if((data[i+j] & 192) != 128) return false; // 192(11000000), 128(10000000)
			}
			i=i+numberOfBytes-1;
		}
		return isValid;
	}

 * 2018/1/29
 * 19. Remove Nth Node From End of List
 * 
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if(head == null || n < 1)return head;
    	ListNode pre = head;
        ListNode cur = head;
        for(int i = 0;i < n;i++){
        	if(cur == null)return head;
        	cur = cur.next;
        }
        if(cur == null){
        	head = head.next;
        	return head;
        }
        while(cur.next != null){
        	cur = cur.next;
        	pre = pre.next;
        }
        pre.next = pre.next.next;
        return head;
    }

 * 2018/1/29
 * 147. Insertion Sort List
 * 
class Solution {
    public ListNode insertionSortList(ListNode head) {
        if(head == null)return head;
        ListNode res = new ListNode(head.val);
        ListNode after;
        ListNode cur = head.next;
        while(cur != null){
        	after = cur.next;
        	res = help(res, cur);
        	cur = after;
        }
        return res;
    }
    
    public ListNode help(ListNode res, ListNode cur){
    	if(cur.val < res.val){
    		cur.next = res;
    		return cur;
    	}
    	ListNode temp = res;
    	while(temp.next != null){
    		if(cur.val < temp.next.val){
    			cur.next = temp.next;
    			temp.next = cur;
    			return res;
    		}
    		temp = temp.next;
    	}
    	temp.next = cur;
    	cur.next = null;
    	return res;
    }
}
 */

/**
 * 2018/1/30
 * 201. Bitwise AND of Numbers Range
 * 
    public int rangeBitwiseAnd(int m, int n) {
        if(m == 0){
            return 0;
        }
        int moveFactor = 1;
        while(m != n){
            m >>= 1;
            n >>= 1;
            moveFactor <<= 1;
        }
        return m * moveFactor;
    }
 */

/**
 * 2018/1/31
 * 117. Populating Next Right Pointers in Each Node II
 * 
    public void connect(TreeLinkNode root) {
        if(root == null)return;
        TreeLinkNode cur = null;
        TreeLinkNode temp = null;
        TreeLinkNode pre = root;
        while(pre != null){
        	if(pre.left == null && pre.right == null){
        		pre = pre.next;
        		continue;
        	}
        	else if(pre.left != null){
        		temp = pre.left;
        		cur = pre.left;
        		if(pre.right != null){
        			cur.next = pre.right;
        			cur = cur.next;
        		}
        	}
        	else{
        		temp = pre.right;
        		cur = pre.right;
        	}
        	while(pre.next != null){
        		pre = pre.next;
        		if(pre.left != null){
        			cur.next = pre.left;
        			cur = cur.next;
        		}
        		if(pre.right != null){
        			cur.next = pre.right;
        			cur = cur.next;
        		}
        	}
        	pre = temp;
        }
    }
 */

/**
 * 2018/2/3
 * 743. Network Delay Time
 * 
    public int networkDelayTime(int[][] times, int N, int K) {//应用迪杰斯特拉算法
    	if(times == null || times.length == 0)return -1;
        int[] res = new int[N + 1];
        Arrays.fill(res, Integer.MAX_VALUE);
        res[K] = 0;
        Set<Integer> in = new HashSet<>();
        int index = K;//最小距离
        int[][] delay = new int[N + 1][N + 1];
        for(int i = 1;i <= N;i++){
        	Arrays.fill(delay[i], Integer.MAX_VALUE);
        	delay[i][i] = 0;
        }
        for(int[] time : times){
        	delay[time[0]][time[1]] = Math.min(time[2], delay[time[0]][time[1]]);
        }
        while(index != N + 1){
        	int cur = index;
        	in.add(cur);
        	for(int i = 1;i <= N;i++){
        		if(!in.contains(i) && delay[cur][i] != Integer.MAX_VALUE){
        			res[i] = Math.min(res[i], res[cur] + delay[cur][i]);
        		}
        	}
        	index = N + 1;
        	int min = Integer.MAX_VALUE;
        	for(int i = 1;i <= N;i++){
        		if(!in.contains(i)){
        			if(min > res[i]){
        				min = res[i];
        				index = i;
        			}
        		}
        	}
        }
        int maxDelay = Integer.MIN_VALUE;
        for(int i = 1;i <= N;i++){
        	if(res[i] == Integer.MAX_VALUE)return -1;
        	else{
        		maxDelay = Math.max(maxDelay, res[i]);
        	}
        }
        return maxDelay;
    }
    
 * 2018/2/3
 * 113. Path Sum II
 * 
class Solution {
	List<List<Integer>> res;
	
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        res = new ArrayList<>();
        if(root == null)return res;
        help(root, sum, new ArrayList<>());
        return res;
    }
    
    public void help(TreeNode root, int sum, List<Integer> list){
    	list.add(root.val);
    	if(root.left == null && root.right == null){
    		if(root.val == sum){
    			List<Integer> temp = new ArrayList<>(list);
    			res.add(temp);
    		}
    	}
    	else{
    		if(root.left != null){
    			help(root.left, sum - root.val, list);
    		}
    		if(root.right != null){
    			help(root.right, sum - root.val, list);
    		}
    	}
    	list.remove(list.size() - 1);
    }
}

 * 2018/2/3
 * 47. Permutations II
 * 
class Solution {
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        boolean[] use = new boolean[nums.length];
        if(nums == null || nums.length == 0)return res;
        Arrays.sort(nums);//很重要，使得相同元素在一起
        dfs(res, nums, use, new ArrayList<>());
        return res;
    }
    
    public void dfs(List<List<Integer>> res, int[] nums, boolean[] use, List<Integer> list){
    	if(list.size() == nums.length){
    		List<Integer> temp = new ArrayList<>(list);
    		res.add(temp);
    		return;
    	}
    	for(int i = 0;i < nums.length;i++){
    		if(use[i])continue;
    		if(i > 0 && nums[i - 1] == nums[i] && !use[i - 1])continue;
    		use[i] = true;
    		list.add(nums[i]);
    		dfs(res, nums, use, list);
    		use[i] = false;
    		list.remove(list.size() - 1);
    	}
    }
}
 */
package exercise;
