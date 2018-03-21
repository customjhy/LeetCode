/**
 * @author jhy code from 2.4 to 2.25
 * 33 questions
 */

/**
 * 2018/2/4
 * 396. Rotate Function
 * 
    public int maxRotateFunction(int[] A) {
        if(A.length <= 1)return 0;
        int sum = 0;
        int cur = 0;
        for(int i = 0;i < A.length;i++){
        	sum += A[i];
        	cur += i * A[i];
        }
        int max = cur;
        int pre = 0;
        int len = A.length;
        for(int i = 0;i < A.length - 1;i++){
        	pre = cur;
        	cur = pre + sum - len * A[len - 1 - i];
        	max = Math.max(max, cur);
        }
        return max;
    }
 */
    
/**
 * 2018/2/5
 * 207. Course Schedule
 * 
	int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
		int left = Math.max(A, E), right = Math.max(Math.min(C, G), left);
		int bottom = Math.max(B, F), top = Math.max(Math.min(D, H), bottom);
		return (C - A) * (D - B) - (right - left) * (top - bottom) + (G - E) * (H - F);
	}

 * 2018/2/5
 * 368. Largest Divisible Subset
 * 
    public List<Integer> largestDivisibleSubset(int[] nums) {
    	if(nums == null || nums.length == 0)return new ArrayList<>();
    	List<Integer> res = new ArrayList<>();
    	int[] dp = new int[nums.length];
    	Arrays.sort(nums);
    	Arrays.fill(dp, 1);//赋初值为1，因其自身可做结果
    	//dp求解最大长度
    	for(int i = 1;i < nums.length;i++){
    		for(int j = i - 1;j >= 0;j--){
    			if(nums[i] % nums[j] == 0){
    				dp[i] = Math.max(dp[i], dp[j] + 1);
    			}
    		}
    	}
    	int maxLen = 1;
    	int index = 0;
    	//求获最大长度的位置
    	for(int i = 1;i < nums.length;i++){
    		if(dp[i] > maxLen){
    			maxLen = dp[i];
    			index = i;
    		}
    	}
    	for(int i = index;i >= 0;i--){
    		if(nums[index] % nums[i] == 0 && dp[i] == maxLen){
    			res.add(nums[i]);
    			maxLen--;
    		}
    	}
    	return res;
    }

 * 2018/2/5
 * 207. Course Schedule
 * 
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        boolean[][] res = new boolean[numCourses][numCourses];
        for(int[] req : prerequisites){
        	res[req[0]][req[1]] = true;
        	for(int i = 0;i < numCourses;i++){
        		if(res[i][req[0]] == true){
        			res[i][req[1]] = true;
        		}
        	}
        }
        for(int[] req : prerequisites){
        	res[req[0]][req[1]] = true;
        	for(int i = 0;i < numCourses;i++){
        		if(res[i][req[0]] == true){
        			res[i][req[1]] = true;
        		}
        	}
        }
        for(int i = 0;i < numCourses;i++){
        	if(res[i][i] == true){
        		return false;
        	}
        }
        return true;
    }
 */

/**
 * 2018/2/6
 * 105. Construct Binary Tree from Preorder and Inorder Traversal
 * 
class Solution {
	public TreeNode buildTree(int[] preorder, int[] inorder) {
		return helper(0, 0, inorder.length - 1, preorder, inorder);
	}

	public TreeNode helper(int preStart, int inStart, int inEnd, int[] preorder, int[] inorder) {
		if (preStart > preorder.length - 1 || inStart > inEnd) {
			return null;
		}
		TreeNode root = new TreeNode(preorder[preStart]);
		int inIndex = 0; // Index of current root in inorder
		for (int i = inStart; i <= inEnd; i++) {
			if (inorder[i] == root.val) {
				inIndex = i;
			}
		}
		root.left = helper(preStart + 1, inStart, inIndex - 1, preorder, inorder);
		root.right = helper(preStart + inIndex - inStart + 1, inIndex + 1, inEnd, preorder, inorder);
		return root;
	}
}

 * 2018/2/6
 * 86. Partition List
 * 
    public ListNode partition(ListNode head, int x) {
        if(head == null)return head;
    	ListNode less = null;
        ListNode more = null;
        ListNode tempL = null;
        ListNode tempM = null;
        while(head != null){
        	if(head.val < x){
        		if(less == null){
        			less = head;
        			tempL = less;
        		}
        		else{
        			tempL.next = head;
        			tempL = tempL.next;
        		}
        	}
        	else{
        		if(more == null){
        			more = head;
        			tempM = more;
        		}
        		else{
        			tempM.next = head;
        			tempM = tempM.next;
        		}
        	}
        	head = head.next;
        }
        if(less == null){
        	return more;
        }
        else if(more == null){
        	return less;
        }
        else{
        	tempM.next = null;
        	tempL.next = more;
        	return less;
        }
    }
    
 * 2018/2/6
 * 417. Pacific Atlantic Water Flow
 * 
class Solution {
    public List<int[]> pacificAtlantic(int[][] matrix) {
        List<int[]> res = new ArrayList<>();
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0)return res;
        int n = matrix.length;
        int m = matrix[0].length;
        boolean[][] pacific = new boolean[n][m];
        boolean[][] atlantic = new boolean[n][m];
        for(int i = 0;i < n;i++){
        	dfs(matrix, pacific, Integer.MIN_VALUE, i, 0);
        	dfs(matrix, atlantic, Integer.MIN_VALUE, i, m - 1);
        }
        for(int i = 0;i < m;i++){
        	dfs(matrix, pacific, Integer.MIN_VALUE, 0, i);
        	dfs(matrix, atlantic, Integer.MIN_VALUE, n - 1, i);
        }
        for(int i = 0;i < n;i++){
        	for(int j = 0;j < m;j++){
        		if(pacific[i][j] && atlantic[i][j]){
        			res.add(new int[]{i,j});
        		}
        	}
        }
        return res;
    }
    
    int[][] dir = new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
    
    public void dfs(int[][] matrix, boolean[][] visit, int height, int x, int y){
    	int n = matrix.length;
    	int m = matrix[0].length;
    	if(x < 0 || y < 0 || x >= n || y >= m || visit[x][y] || height > matrix[x][y])return;
    	visit[x][y] = true;
    	for(int[] d : dir){
    		dfs(matrix, visit, matrix[x][y], x + d[0], y + d[1]);
    	}
    }
}
    
 * 2018/2/6
 * 187. Repeated DNA Sequences
 * 
    public List<String> findRepeatedDnaSequences(String s) {
    	if(s.length() < 10)return new ArrayList<>();
    	Set<String> set = new HashSet<>();
    	Set<String> resSet = new HashSet<>();
        for(int i = 0;i <= s.length() - 10;i++){
        	String temp = s.substring(i, i + 10);
        	if(set.contains(temp)){
        		resSet.add(temp);
        	}
        	else{
        		set.add(temp);
        	}
        }
        return new ArrayList<>(resSet);
    }
 */

/**
 * 2018/2/7
 * 542. 01 Matrix
 * 
	public int[][] updateMatrix(int[][] matrix) {
		if (matrix.length == 0 || matrix[0].length == 0) {
			return matrix;
		}
		int[][] dis = new int[matrix.length][matrix[0].length];
		int range = matrix.length * matrix[0].length;

		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {
				if (matrix[i][j] == 0) {
					dis[i][j] = 0;
				} else {
					int upCell = (i > 0) ? dis[i - 1][j] : range;
					int leftCell = (j > 0) ? dis[i][j - 1] : range;
					dis[i][j] = Math.min(upCell, leftCell) + 1;
				}
			}
		}

		for (int i = matrix.length - 1; i >= 0; i--) {
			for (int j = matrix[0].length - 1; j >= 0; j--) {
				if (matrix[i][j] == 0) {
					dis[i][j] = 0;
				} else {
					int downCell = (i < matrix.length - 1) ? dis[i + 1][j] : range;
					int rightCell = (j < matrix[0].length - 1) ? dis[i][j + 1] : range;
					dis[i][j] = Math.min(Math.min(downCell, rightCell) + 1, dis[i][j]);
				}
			}
		}

		return dis;
	}
	
 * 2018/2/7
 * 106. Construct Binary Tree from Inorder and Postorder Traversal
 * 
class Solution {
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        if(inorder == null || inorder.length == 0)return null;
        return help(inorder, postorder, 0, inorder.length - 1, inorder.length - 1);
    }
    
    public TreeNode help(int[] inorder, int[] postorder, int start, int end, int post){//start,end为inorder坐标
    	if(start > end || post < 0)return null;
    	int index = start;
    	for(int i = start;i <= end;i++){
    		if(inorder[i] == postorder[post]){
    			index = i;
    			break;
    		}
    	}
    	TreeNode root = new TreeNode(postorder[post]);
    	root.left = help(inorder, postorder, start, index - 1, post - end + index - 1);
    	root.right = help(inorder, postorder, index + 1, end, post - 1);
    	return root;
    }
}

 * 2018/2/7
 * 713. Subarray Product Less Than K
 * 
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        int n = nums.length;
        long sum = 1l;
        int left = 0;
        int right = 0;
        int total = 0;
        while(right < n){
            sum *= nums[right];
            while(left <= right&&sum >= k){
                sum /= nums[left];
                left++;
            }
            total += (right - left + 1);
            right++;
        }
        return total;
    }
 */

/**
 * 2018/2/8
 * 63. Unique Paths II
 * 
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if(obstacleGrid == null || obstacleGrid.length == 0 || obstacleGrid[0].length == 0)return 0;
        int n = obstacleGrid.length;
        int m = obstacleGrid[0].length;
        int[][] dp = new int[n][m];
        for(int i = 0;i < m;i++){
        	if(obstacleGrid[0][i] == 0){
        		dp[0][i] = 1;
        	}
        	else break;
        }
        for(int i = 0;i < n;i++){
        	if(obstacleGrid[i][0] == 0){
        		dp[i][0] = 1;
        	}
        	else break;
        }
        for(int i = 1;i < n;i++){
        	for(int j = 1;j < m;j++){
        		if(obstacleGrid[i][j] == 0){
            		if(obstacleGrid[i - 1][j] == 0) dp[i][j] += dp[i - 1][j];
            		if(obstacleGrid[i][j - 1] == 0) dp[i][j] += dp[i][j - 1];
        		}
        	}
        }
        return dp[n - 1][m - 1];
    }
    
 * 2018/2/8
 * 779. K-th Symbol in Grammar
 * 
    public int kthGrammar(int N, int K) {
        if(N == 1)return 0;
        if(K % 2 == 0)return kthGrammar(N - 1, K / 2) == 0 ? 1 : 0;
        else return kthGrammar(N - 1, (K + 1) / 2) == 0 ? 0 : 1;
    }
    
 * 2018/2/8
 * 209. Minimum Size Subarray Sum
 * 
    public int minSubArrayLen(int s, int[] nums) {
        if(nums == null || nums.length == 0)return 0;
        int res = Integer.MAX_VALUE;
        int left = 0;
        int right = 0;
        int sum = 0;
        while(right < nums.length && left < nums.length){
        	if(sum < s){
        		sum += nums[right];
        		right++;
        	}
        	else{
        		res = Math.min(res, right - left);
        		sum -= nums[left];
        		left++;
        	}
        }
        while(sum >= s){
    		res = Math.min(res, right - left);
    		sum -= nums[left];
    		left++;
        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }
 */

/**
 * 2018/2/11
 * 783. Minimum Distance Between BST Nodes
 * 
class Solution {
    public int minDiffInBST(TreeNode root) {
        List<Integer> nums = new ArrayList<>();
        if(root == null)return -1;
        dfs(root, nums);
        int res = Integer.MAX_VALUE;
        for(int i = 0;i < nums.size() - 1;i++){
        	res = Math.min(res, nums.get(i + 1) - nums.get(i));
        }
        return res;
    }
    
    public void dfs(TreeNode root, List<Integer> nums){
    	if(root.left != null)dfs(root.left, nums);
    	nums.add(root.val);
    	if(root.right != null)dfs(root.right, nums);
    }
}
 */

/**
 * 2018/2/16
 * 467. Unique Substrings in Wraparound String
 * 
	public int findSubstringInWraproundString(String p) {
		// count[i] is the maximum unique substring end with ith letter.
		// 0 - 'a', 1 - 'b', ..., 25 - 'z'.
		int[] count = new int[26];

		// store longest contiguous substring ends at current position.
		int maxLengthCur = 0;

		for (int i = 0; i < p.length(); i++) {
			if (i > 0 && (p.charAt(i) - p.charAt(i - 1) == 1 || (p.charAt(i - 1) - p.charAt(i) == 25))) {
				maxLengthCur++;
			} else {
				maxLengthCur = 1;
			}

			int index = p.charAt(i) - 'a';
			count[index] = Math.max(count[index], maxLengthCur);
		}

		// Sum to get result
		int sum = 0;
		for (int i = 0; i < 26; i++) {
			sum += count[i];
		}
		return sum;
	}

 * 2018/2/16
 * 33. Search in Rotated Sorted Array
 * 
    public int search(int[] nums, int target) {
    	int low = 0;
    	int high = nums.length - 1;
    	while(low < high){
    		int mid = (low + high) / 2;
    		if(nums[mid] < nums[high]){
    			low = mid + 1;
    		}
    		else{
    			high = mid;
    		}
    	}
    	int center = low;
    	low = 0;
    	high = nums.length - 1;
    	while(low <= high){
    		int mid = (low + high) / 2;
    		int realMid = (mid + center) % nums.length;
    		if(nums[realMid] == target)return realMid;
    		else if(nums[realMid] > target){
    			high = mid - 1;
    		}
    		else{
    			low = mid + 1;
    		}
    	}
    	return -1;
    }
 */

/**
 * 2018/2/17
 * 781. Rabbits in Forest
 * 
    public int numRabbits(int[] answers) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int answer : answers){
        	map.put(answer, map.getOrDefault(answer, 0) + 1);
        }
        int sum = 0;
        for(int num : map.keySet()){
        	int time = map.get(num);
        	while(time > 0){
        		sum += (num + 1);
        		time -= (num + 1);
        	}
        }
        return sum;
    }
 */

/**
 * 2018/2/18
 * 95. Unique Binary Search Trees II
 * 
class Solution {
    public List<TreeNode> generateTrees(int n) {
        List<List<TreeNode>> dp = new ArrayList<>();
        List<TreeNode> temp = new ArrayList<>();
        if(n == 0)return temp;
        temp.add(new TreeNode(1));
        if(n == 1)return temp;
        dp.add(temp);
        for(int i = 2;i <= n;i++){
        	temp = new ArrayList<>();
        	for(int j = 1;j <= i;j++){
        		for(TreeNode left : leftGenerate(dp, j - 1)){
        			for(TreeNode right : rightGenerate(dp, i - j, j)){
        				TreeNode root = new TreeNode(j);
        				root.left = left;
        				root.right = right;
        				temp.add(root);
        			}
        		}
        	}
        	dp.add(temp);
        }
        return dp.get(n - 1);
    }
    
    public List<TreeNode> leftGenerate(List<List<TreeNode>> dp, int root){
    	if(root == 0){
    		List<TreeNode> res = new ArrayList<>();
    		res.add(null);
    		return res;
    	}
    	List<TreeNode> res = dp.get(root - 1);
    	return new ArrayList<>(res);
    }
    
    public List<TreeNode> rightGenerate(List<List<TreeNode>> dp, int root, int val){
    	if(root == 0){
    		List<TreeNode> res = new ArrayList<>();
    		res.add(null);
    		return res;
    	}
    	List<TreeNode> nodes = dp.get(root - 1);
    	List<TreeNode> res = new ArrayList<>();
    	for(TreeNode node : nodes){
    		res.add(clone(node, val));
    	}
    	return res;
    }
    
    private static TreeNode clone(TreeNode n, int offset) {
        if (n == null) {
            return null;
        }
        TreeNode node = new TreeNode(n.val + offset);
        node.left = clone(n.left, offset);
        node.right = clone(n.right, offset);
        return node;
    }
}
 */

/**
 * 2018/2/20
 * 784. Letter Case Permutation
 * 
class Solution {
	public List<String> letterCasePermutation(String S) {
		List<String> res = new ArrayList<>();
		if (S.length() == 0)
			return res;
		help(res, S.toLowerCase().toCharArray(), 0);
		return res;
	}

	public int Nearindex(char[] str, int start) {//找到最近字母的索引
		while (start < str.length && !Character.isLetter(str[start])) {
			start++;
		}
		return start;
	}

	public void help(List<String> res, char[] str, int start) {
		if (start == str.length) {
			res.add(new String(str));
			return;
		}
		help(res, str, Nearindex(str, start + 1));
		if (Character.isLetter(str[start])) {
			str[start] -= 32;
			help(res, str, Nearindex(str, start + 1));
			str[start] += 32;
		}
	}
}
 */

/**
 * 2018/2/21
 * 785. Is Graph Bipartite?
 * 
    public boolean isBipartite(int[][] graph) {
        Set<Integer> left = new HashSet<>();
        Set<Integer> right = new HashSet<>();
        boolean[] visit = new boolean[graph.length];
        Queue<Integer> queue = new LinkedList<>();
        for(int i = 0;i < graph.length;i++){
        	if(!visit[i]){
        		queue.add(i);
        		while(!queue.isEmpty()){
        			int temp = queue.poll();
        			visit[temp] = true;
        			if(left.contains(temp)){//如果左集合包含temp，则将其他点加入到右集合
        				for(int j = 0;j < graph[temp].length;j++){
        					if(left.contains(graph[temp][j]))return false;
        					right.add(graph[temp][j]);
        					if(!visit[graph[temp][j]]){
        						queue.add(graph[temp][j]);
        					}
        				}
        			}
        			else{//如果左集合不包含temp，则将temp加入右集合
        				right.add(temp);
        				for(int j = 0;j < graph[temp].length;j++){
        					if(right.contains(graph[temp][j]))return false;
        					left.add(graph[temp][j]);
        					if(!visit[graph[temp][j]]){
        						queue.add(graph[temp][j]);
        					}
        				}
        			}
        		}
        	}
        }
    	return true;
    }
}

 * 2018/2/21
 * 34. Search for a Range
 * 
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int index = binarySearch(nums, target, 0, nums.length - 1);
        int left = index;
        int right = index;
        while((index = binarySearch(nums, target, 0, left - 1)) >= 0){
        	left = index;
        }
        while((index = binarySearch(nums, target, right + 1, nums.length - 1)) >= 0){
        	right = index;
        }
        return new int[]{left,right};
    }
    
	// 递归实现二分查找
	public static int binarySearch(int[] dataset, int data, int beginIndex, int endIndex) {
		int midIndex = (beginIndex + endIndex) / 2;
		if (beginIndex > endIndex || beginIndex < 0 || endIndex >= dataset.length || data < dataset[beginIndex] || data > dataset[endIndex]) {
			return -1;
		}
		if (data < dataset[midIndex]) {
			return binarySearch(dataset, data, beginIndex, midIndex - 1);
		} else if (data > dataset[midIndex]) {
			return binarySearch(dataset, data, midIndex + 1, endIndex);
		} else {
			return midIndex;
		}
	}
}
 */

/**
 * 2018/2/22
 * 81. Search in Rotated Sorted Array II
 * 
    public boolean search(int[] nums, int target) {
        int start = 0;
        int end = nums.length - 1;
        int mid = -1;
        while(start <= end){
        	mid = (start + end) / 2;
        	if(nums[mid] == target)return true;
        	//if right is sorted or left is unsorted
        	if(nums[mid] < nums[end] || nums[mid] < nums[start]){
        		if(nums[mid] < target && target <= nums[end]){
        			start = mid + 1;
        		}
        		else{
        			end = mid - 1;
        		}
        	}
        	//if left is sorted or right is unsorted
        	else if(nums[mid] > nums[start] || nums[mid] > nums[end]){
        		if(nums[start] <= target && target < nums[mid]){
        			end = mid - 1;
        		}
        		else {
					start = mid + 1;
				}
        	}
        	//if nums[mid] == nums[start] or nums[mid] == nums[end]
        	else{
        		end--;
        	}
        }
        return false;
    }
 */

/**
 * 2018/2/23
 * 522. Longest Uncommon Subsequence II
 * 
class Solution {
    public int findLUSlength(String[] strs) {
    	Arrays.sort(strs , new Comparator<String>() {
			public int compare(String o1, String o2) {
				if(o1.length() != o2.length()){
					return o2.length() - o1.length();
				}
				else{
					return o1.compareTo(o2);
				}
			}
		});
    	for(int i = 0;i < strs.length;i++){
    		int j = i + 1;
    		while(j < strs.length && strs[j].length() == strs[i].length()){
    			j++;
    		}
    		boolean flag = true;
    		for(int k = 0;k < j;k++){
    			if(k == i)continue;
    			if(isSubsequence(strs[i], strs[k])){
    				flag = false;
    				break;
    			}
    		}
    		if(flag)return strs[i].length();
    	}
    	return -1;
    }
    
    public boolean isSubsequence(String small, String big) {
		int i = 0;
    	for(int j = 0;j < big.length();j++){
    		if(small.charAt(i) == big.charAt(j)){
    			i++;
    			if(i == small.length())return true;
    		}
    	}
    	return false;
	}
}

 * 2018/2/23
 * 787. Cheapest Flights Within K Stops
 * 
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
        int[] distance = new int[n];
        Arrays.fill(distance, Integer.MAX_VALUE);
        distance[src] = 0;
        Queue<Integer> queue = new LinkedList<>();
        queue.add(src);
        while(K-- >= 0 && !queue.isEmpty()){
        	int size = queue.size();
        	for(int i = 0;i < size;i++){
        		int cur = queue.poll();
        		for(int[] f : flights){
        			if(f[0] == cur && distance[f[1]] > distance[f[0]] + f[2]){
        				distance[f[1]] = distance[f[0]] + f[2];
        				queue.add(f[1]);
        			}
        		}
        	}
        }
        return distance[dst] == Integer.MAX_VALUE ? -1 : distance[dst];
    }
 */

/**
 * 2018/2/24
 * 228. Summary Ranges
 * 
    public List<String> summaryRanges(int[] nums) {
        List<String> res = new ArrayList<>();
        if(nums == null || nums.length == 0)return res;
        for(int i = 0; i < nums.length;){
        	int j = i + 1;
        	while(j < nums.length && nums[j] == nums[j - 1] + 1){
        		j++;
        	}
        	if(i == j - 1){
        		String temp = Integer.toString(nums[i]);
        		res.add(temp);
        	}
        	else{
        		String temp = Integer.toString(nums[i]) + "->" + Integer.toString(nums[j - 1]);
        		res.add(temp);
        	}
        	i = j;
        }
        return res;
    }
    
 * 2018/2/24
 * 56. Merge Intervals
 * 
    public List<Interval> merge(List<Interval> intervals) {
    	List<Interval> res = new ArrayList<>();
    	if(intervals == null || intervals.size() == 0)return res;
    	if(intervals.size() == 1)return intervals;
        Collections.sort(intervals, new Comparator<Interval>() {
			public int compare(Interval o1, Interval o2) {
				if(o1.start != o2.start){
					return o1.start - o2.start;
				}
				return o1.end - o2.end;
			}
		});
    	res.add(intervals.get(0));
    	for(int i = 1;i < intervals.size();i++){
    		Interval temp = intervals.get(i);
    		Interval last = res.get(res.size() - 1);
    		if(temp.start <= last.end){
    			res.remove(res.size() - 1);
    			res.add(new Interval(last.start, Math.max(last.end, temp.end)));
    		}
    		else{
    			res.add(temp);
    		}
    	}
        return res;
    }
    
 * 2018/2/24
 * 139. Word Break
 * 
    public boolean wordBreak(String s, List<String> wordDict) {
        boolean[] res = new boolean[s.length()];
        Set<String> dict = new HashSet<>();
        for(String word : wordDict)
        {
        	dict.add(word);
        }
    	for(int i = 0;i < s.length();i++){
    		for(int j = i - 1;j >= 0;j--){
    			if(res[j] && dict.contains(s.substring(j + 1, i + 1))){
    				res[i] = true;
    				break;
    			}
    		}
    		if(res[i] || dict.contains(s.substring(0, i + 1))){
    			res[i] = true;
    		}
    	}
    	return res[s.length() - 1];
    }
 */

/**
 * 2018/2/25
 * 221. Maximal Square
 * 
class Solution {
    public int maximalSquare(char[][] matrix) {
    	if(matrix == null || matrix.length == 0)return 0;
    	if(matrix[0] == null || matrix[0].length == 0)return 0;
    	int row = matrix.length;
    	int col = matrix[0].length;
    	int res = 0;
    	int[][] dp = new int[row][col];
    	//初始化dp
    	for(int i = 0;i < row;i++){
    		if(matrix[i][0] == '1'){
    			res = 1;
    			dp[i][0] = 1;
    		}
    	}
    	for(int i = 0;i < col;i++){
    		if(matrix[0][i] == '1'){
    			res = 1;
    			dp[0][i] = 1;
    		}
    	}
		for (int i = 1; i < row; i++) {
			for (int j = 1; j < col; j++) {
				dp[i][j] = squareNum(matrix, i, j, dp[i - 1][j - 1]);
				res = Math.max(res, dp[i][j]);
			}
		}
		return res * res;
	}
    
    public int squareNum(char[][] matrix, int i, int j,int num){
    	if(matrix[i][j] == '0')return 0;
    	for(int k = 1;k <= num;k++){
    		if(i - k < 0 || j - k < 0 || matrix[i - k][j] == '0' || matrix[i][j - k] == '0'){
    			return k;
    		}
    	}
    	return num + 1;
    }
}

 * 2018/2/25
 * 788. Rotated Digits
 * 
class Solution {
    public int rotatedDigits(int N) {
        int res = 0;
        Set<Integer> equal = new HashSet<>();
        equal.add(0);
        equal.add(1);
        equal.add(8);
        Set<Integer> valid = new HashSet<>();
        valid.add(2);
        valid.add(5);
        valid.add(6);
        valid.add(9);
        for(int i = 1;i <= N;i++){
        	if(isValid(i, equal, valid))res++;
        }
        return res;
    }
    
    public boolean isValid(int n, Set<Integer> equal, Set<Integer> valid){
    	boolean flag = false;
    	while(n > 0){
    		int temp = n % 10;
    		n /= 10;
    		if(valid.contains(temp))flag = true;
    		else if(equal.contains(temp)) continue;
    		else return false;
    	}
    	return flag;
    }
}

 * 2018/2/25
 * 789. Escape The Ghosts
 * 
	public boolean escapeGhosts(int[][] ghosts, int[] target) {
		int count = Math.abs(target[0]) + Math.abs(target[1]);
		for (int[] ghost : ghosts) {
			if (Math.abs(target[0] - ghost[0]) + Math.abs(target[1] - ghost[1]) <= count)
				return false;
		}
		return true;
	}

 * 2018/2/25
 * 790. Domino and Tromino Tiling
 * 
    public int numTilings(int N) {
    	if(N == 1)return 1;
        if(N == 2)return 2;
        long[] dp = new long[N + 1];
        dp[0] = 1;
        dp[1] = 1;
        dp[2] = 2;
        for(int i = 3;i <= N;i++){
        	dp[i] = 2 * dp[i - 1] + dp[i - 3];
        	dp[i] %= 1000000007;
        }
        return (int)dp[N];
    }
    
 * 2018/2/25
 * 791. Custom Sort String
 * 
    public String customSortString(String S, String T) {
        int[] times = new int[26];
        for(char ch : T.toCharArray()){
        	times[ch - 'a']++;
        }
        StringBuffer res = new StringBuffer();
        for(char ch : S.toCharArray()){
        	while(times[ch - 'a'] > 0){
        		res.append(ch);
        		times[ch - 'a']--;
        	}
        }
        for(int i = 0;i < 26;i++){
        	if(times[i] != 0){
        		char temp = (char)('a' + i);
        		for(int j = 0;j < times[i];j++){
        			res.append(temp);
        		}
        	}
        }
        return res.toString();
    }
 */
package exercise;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

class Solution_0204_To_0225 {

}
