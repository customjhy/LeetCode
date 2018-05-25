/**
 * @author jhy code from 5.7 to 5.19
 * 25 questions
 */

/**
 * 2018/5/7
 * 57. Insert Interval
 * 
    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
    	List<Interval> res = new ArrayList<>();
    	if(newInterval.start > newInterval.end)return intervals;
    	if(intervals.size() == 0){
    		res.add(newInterval);
    		return res;
    	}
    	if(newInterval.end < intervals.get(0).start){
    		intervals.add(0, newInterval);
    	}else if(newInterval.start > intervals.get(intervals.size() - 1).end){
    		intervals.add(newInterval);
    	}
    	int left = 0;
    	while(left < intervals.size() && newInterval.start > intervals.get(left).end)
    		left++;
    	int right = intervals.size() - 1;
    	while(right >= 0 && newInterval.end < intervals.get(right).start)
    		right--;
    	for(int i = 0;i < left;i++){
    		res.add(intervals.get(i));
    	}
    	res.add(new Interval(Math.min(intervals.get(left).start, newInterval.start), Math.max(intervals.get(right).end, newInterval.end)));
    	for(int i = right + 1;i < intervals.size();i++){
    		res.add(intervals.get(i));
    	}
    	return res;
    }
    
 * 2018/5/7
 * 233. Number of Digit One
 * 
    public int countDigitOne(int n) {
        int res = 0;
        for(long i = 1;i <= n;i *= 10){
        	long divide = i * 10;
        	res += (n / divide) * i + Math.min(Math.max(n % divide - i + 1, 0), i);
        }
        return res;
    }
    
 * 2018/5/7
 * 23. Merge k Sorted Lists
 * 
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
    	if(lists == null || lists.length == 0)return null;
        List<ListNode> list1 = new ArrayList<>();
        for(ListNode ln : lists)list1.add(ln);
        while(list1.size() > 1){
        	List<ListNode> tempList = new ArrayList<>();
        	int i = 0;
        	for(i = 0;i < list1.size() - 1;i += 2){
        		tempList.add(mergeTwoLists(list1.get(i), list1.get(i + 1)));
        	}
        	if(i == list1.size() - 1)tempList.add(list1.get(i));
        	list1 = tempList;
        }
        return list1.get(0);
    }
    
    public ListNode mergeTwoLists(ListNode list1, ListNode list2){
    	ListNode head = new ListNode(0);
    	ListNode temp = head;
    	while(list1 != null && list2 != null){
    		if(list1.val < list2.val){
    			temp.next = list1;
    			temp = temp.next;
    			list1 = list1.next;
    		}else{
    			temp.next = list2;
    			temp = temp.next;
    			list2 = list2.next;
    		}
    	}
    	if(list1 == null)
    		temp.next = list2;
    	else
    		temp.next = list1;
    	return head.next;
    }
}

 * 2018/5/7
 * 446. Arithmetic Slices II - Subsequence
 * 
    public int numberOfArithmeticSlices(int[] A) {
        int res = 0;
        int n = A.length;
		Map<Integer, Integer>[] maps = new Map[n];
        for(int i = 0;i < A.length;i++){
        	maps[i] = new HashMap<>();
        	for(int j = 0;j < i;j++){
        		long delta = (long)A[i] - (long)A[j];
                if (delta < Integer.MIN_VALUE || delta > Integer.MAX_VALUE) {
                    continue;
                }
                int diff = (int)delta;
                int sum = maps[j].getOrDefault(diff, 0);
                int ori = maps[i].getOrDefault(diff, 0);
                res += sum;
                maps[i].put(diff, sum + ori + 1);
        	}
        }
        return res;
    }
*/

/**
 * 2018/5/8
 * 84. Largest Rectangle in Histogram
 * 
    public int largestRectangleArea(int[] heights) {
        if(heights == null || heights.length == 0)return 0;
    	int N = heights.length;
    	int[] left = new int[N];
    	for(int i = 0;i < N;i++){
    		int j = i;
    		while(j >= 0 && heights[i] <= heights[j]){
    			j--;
    		}
    		left[i] = j + 1;
    	}
    	int[] right = new int[N];
    	for(int i = N - 1;i >= 0;i--){
    		int j = i;
    		while(j < N && heights[i] <= heights[j]){
    			j++;
    		}
    		right[i] = j - 1;
    	}
    	int res = 0;
    	for(int i = 0;i < N;i++){
    		res = Math.max(res, heights[i] * (right[i] - left[i] + 1));
    	}
    	return res;
    }
    
 * 2018/5/8
 * 391. Perfect Rectangle
 * 
    public boolean isRectangleCover(int[][] rectangles) {
        if(rectangles == null || rectangles.length == 0)
        	return false;
        int x1 = Integer.MAX_VALUE;
        int y1 = Integer.MAX_VALUE;
        int x2 = Integer.MIN_VALUE;
        int y2 = Integer.MIN_VALUE;
        int area = 0;
        Set<String> set = new HashSet<>();
        for(int[] rect : rectangles){
        	x1 = Math.min(x1, rect[0]);
        	y1 = Math.min(y1, rect[1]);
        	x2 = Math.max(x2, rect[2]);
        	y2 = Math.max(y2, rect[3]);
        	area += (rect[2] - rect[0]) * (rect[3] - rect[1]);
        	
        	String s1 = rect[0] + " " + rect[1];
        	String s2 = rect[2] + " " + rect[1];
        	String s3 = rect[0] + " " + rect[3];
        	String s4 = rect[2] + " " + rect[3];
        	
        	if(!set.add(s1))set.remove(s1);
        	if(!set.add(s2))set.remove(s2);
        	if(!set.add(s3))set.remove(s3);
        	if(!set.add(s4))set.remove(s4);
        }
    	if(!set.contains(x1 + " " + y1) || !set.contains(x1 + " " + y2) || !set.contains(x2 + " " + y1) || !set.contains(x2 + " " + y2) || set.size() != 4)
    		return false;
        return area == (x2 - x1) * (y2 - y1);
    }
    
 * 2018/5/8
 * 780. Reaching Points
 * 
    public boolean reachingPoints(int sx, int sy, int tx, int ty) {
        if(sx == tx && sy == ty)
        	return true;
        if(tx == ty || sx > tx || sy > ty)
        	return false;
        if(tx > ty){
        	int quotient = Math.max(1, (tx - sx) / ty);
        	return reachingPoints(sx, sy, tx - ty * quotient, ty);
        }else{
        	int quotient = Math.max(1, (ty - sy) / tx);
        	return reachingPoints(sx, sy, tx, ty - tx * quotient);
        }
    }
*/

/**
 * 2018/5/9
 * 4. Median of Two Sorted Arrays
 * 
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        if(m > n){
        	int[] temp = nums1;nums1 = nums2;nums2 = temp;
        	int t = m;m = n;n = t;
        }
        int min = 0;
        int max = m;
        int half = (m + n + 1) / 2;
        while(min <= max){
        	int i = (min + max) / 2;
        	int j = half - i;
        	if(i < max && nums2[j - 1] > nums1[i]){
        		min++;
        	}else if(i > min && nums1[i - 1] > nums2[j]){
        		max--;
        	}else{
        		int maxLeft = 0;
        		if(i == 0)maxLeft = nums2[j - 1];
        		else if(j == 0)maxLeft = nums1[i - 1];
        		else{
        			maxLeft = Math.max(nums1[i - 1], nums2[j - 1]);
        		}
        		if((m + n) % 2 == 1)return maxLeft;
        		
        		int minRight = 0;
        		if(i == m)minRight = nums2[j];
        		else if(j == n)minRight = nums1[i];
        		else{
        			minRight = Math.min(nums1[i], nums2[j]);
        		}
        		return (maxLeft + minRight) / 2.0;
        	}
        }
        return 0.0;
    }
    
 * 2018/5/9
 * 719. Find K-th Smallest Pair Distance
 * 
    Solution1: PriorityQueue--TTL
    public int smallestDistancePair(int[] nums, int k) {
        PriorityQueue<Integer> queue = new PriorityQueue<>((a, b) -> b - a);
        Arrays.sort(nums);
        int i = 0;
        //initial
        for(i = 0;i < nums.length - 1;i++){
        	for(int j = i + 1;j < nums.length;j++){
        		queue.add(nums[j] - nums[i]);
        	}
        	if(queue.size() >= k)break;
        }
        while(queue.size() > k)queue.poll();
        for(i = i + 1;i < nums.length - 1;i++){
        	for(int j = i + 1;j < nums.length;j++){
        		if(nums[j] - nums[i] >= queue.peek()){
        			break;
        		}else{
        			queue.add(nums[j] - nums[i]);
        			queue.poll();
        		}
        	}
        }
        return queue.peek();
    }
    
    Solution2: binary Search + two Pointers
    public int smallestDistancePair(int[] nums, int k) {
        Arrays.sort(nums);
        int low = 0;
        int high = nums[nums.length - 1] - nums[0];
        while(low < high){
        	int mid = (low + high) / 2;
        	int left = 0;
        	int count = 0;
        	for(int right = 0;right < nums.length;right++){
        		while(nums[right] - nums[left] > mid)left++;
        		count += right - left;
        	}
        	if(count >= k)high = mid;
        	else low = mid + 1;
        }
        return low;
    }
*/

/**
 * 2018/5/10
 * 124. Binary Tree Maximum Path Sum
 * 
class Solution {
    public int maxPathSum(TreeNode root) {
        Map<TreeNode, Integer> map = new HashMap<>();
        buildMap(root, map);
        return dfs(root, map);
    }
    
    public int dfs(TreeNode root, Map<TreeNode, Integer> map){
    	if(root == null)return Integer.MIN_VALUE;
    	int left = map.getOrDefault(root.left, 0);
    	int right = map.getOrDefault(root.right, 0);
    	int sum = root.val;
    	if(left > 0)sum += left;
    	if(right > 0)sum += right;
    	return Math.max(sum, Math.max(dfs(root.left, map), dfs(root.right, map)));
    }
    
    public int buildMap(TreeNode root, Map<TreeNode, Integer> map){
    	if(root == null)return 0;
    	if(root.left == null && root.right == null){
    		map.put(root, root.val);
    		return root.val;
    	}
    	int left = buildMap(root.left, map);
    	int right = buildMap(root.right, map);
    	int max = root.val;
    	if(left > 0 || right > 0)
    		max += Math.max(left, right);
    	map.put(root, max);
    	return max;
    }
}

 * 2018/5/10
 * 124. Binary Tree Maximum Path Sum
 * 
class Solution {
	public List<List<Integer>> palindromePairs(String[] words) {
	    List<List<Integer>> ret = new ArrayList<>(); 
	    if (words == null || words.length < 2) return ret;
	    Map<String, Integer> map = new HashMap<String, Integer>();
	    for (int i=0; i<words.length; i++) map.put(words[i], i);
	    for (int i=0; i<words.length; i++) {
	        for (int j=0; j<=words[i].length(); j++) { // notice it should be "j <= words[i].length()"
	            String str1 = words[i].substring(0, j);
	            String str2 = words[i].substring(j);
	            if (isPalindrome(str1)) {
	                String str2rvs = new StringBuilder(str2).reverse().toString();
	                if (map.containsKey(str2rvs) && map.get(str2rvs) != i) {
	                    List<Integer> list = new ArrayList<Integer>();
	                    list.add(map.get(str2rvs));
	                    list.add(i);
	                    ret.add(list);
	                }
	            }
	            if (isPalindrome(str2)) {
	                String str1rvs = new StringBuilder(str1).reverse().toString();
	                // check "str.length() != 0" to avoid duplicates
	                if (map.containsKey(str1rvs) && map.get(str1rvs) != i && str2.length()!=0) { 
	                    List<Integer> list = new ArrayList<Integer>();
	                    list.add(i);
	                    list.add(map.get(str1rvs));
	                    ret.add(list);
	                }
	            }
	        }
	    }
	    return ret;
	}

	private boolean isPalindrome(String str) {
	    int left = 0;
	    int right = str.length() - 1;
	    while (left <= right) {
	        if (str.charAt(left++) !=  str.charAt(right--)) return false;
	    }
	    return true;
	}
}
*/

/**
 * 2018/5/11
 * 45. Jump Game II
 * 
	Solution1: dynamic programming--TTL
    public int jump(int[] nums) {
        int[] dp = new int[nums.length];//minimum step to reach the end
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for(int i = 0;i < nums.length;i++){
        	for(int j = 1;j <= nums[i] && i + j < nums.length;j++){
        		dp[i + j] = Math.min(dp[i + j], dp[i] + 1);
        	}
        }
        return dp[nums.length - 1];
    }
    
	Solution2: greedy--AC
	public int jump(int[] nums) {
        if(nums.length < 2)return 0;
		int level = 0;
		int i = 0;
		int curMax = 0;
		int nextMax = 0;
		while(curMax >= i){
			level++;
			for(;i <= curMax;i++){
				nextMax = Math.max(nextMax, i + nums[i]);
				if(nextMax >= nums.length)return level;
			}
			curMax = nextMax;
		}
		return 0;
	}
	
 * 2018/5/11
 * 466. Count The Repetitions
 * 
    public int getMaxRepetitions(String s1, int n1, String s2, int n2) {
    	if(n1 == 0)return 0;
    	int[] indexs = new int[s2.length() + 1];
    	int[] counts = new int[s2.length() + 1];
    	int count = 0;
    	int index = 0;
    	for(int i = 1;i <= n1;i++){
    		for(int j = 0;j < s1.length();j++){
    			if(s1.charAt(j) == s2.charAt(index)){
    				index++;
    				if(index == s2.length()){
    					index = 0;
    					count++;
    				}
    			}
    		}
    		indexs[i] = index;
    		counts[i] = count;
    		for(int k = 0;k < i;k++){
    			if(indexs[k] == indexs[i]){
    				int pre = counts[k];
    				int cur = (counts[i] - counts[k]) * ((n1 - k) / (i - k));
    				int post = counts[k + (n1 - k) % (i - k)] - counts[k];
    				return (pre + cur + post) / n2;
    			}
    		}
    	}
    	return counts[n1] / n2;
    }
*/

/**
 * 2018/5/12
 * 188. Best Time to Buy and Sell Stock IV
 * 
    public int maxProfit(int k, int[] prices) {
        if(prices == null || prices.length < 2)return 0;
        if(k <= 0)return 0;
        k = Math.min(k, prices.length);
    	int[] buy = new int[k];
    	Arrays.fill(buy, Integer.MIN_VALUE);
        int[] sell = new int[k];
        for(int i = 0;i < prices.length;i++){
        	buy[0] = Math.max(buy[0], 0 - prices[i]);
        	sell[0] = Math.max(sell[0], buy[0] + prices[i]);
        	for(int j = 1;j < k;j++){
        		buy[j] = Math.max(buy[j], sell[j - 1] - prices[i]);
        		sell[j] = Math.max(sell[j], buy[j] + prices[i]);
        	}
        }
        return sell[k -1];
    }
*/

/**
 * 2018/5/13
 * 832. Flipping an Image
 * 
    public int[][] flipAndInvertImage(int[][] A) {
    	if(A == null || A.length == 0 || A[0] == null || A[0].length == 0)return A;
    	int row = A.length;
    	int col = A[0].length;
        int[][] res = new int[row][col];
        for(int i = 0;i < row;i++){
        	for(int j = 0;j < col;j++){
        		if(A[i][j] == 0){
        			res[i][col - j - 1] = 1;
        		}
        	}
        }
        return res;
    }
*/

/**
 * 2018/5/16
 * 833. Find And Replace in String
 * 
class Solution {
    public String findReplaceString(String S, int[] indexes, String[] sources, String[] targets) {
        StringBuilder res = new StringBuilder();
        int n = indexes.length;
        int[][] index = new int[n][2];
        for(int i = 0;i < n;i++){
        	index[i][0] = indexes[i];
        	index[i][1] = i;
        }
        Arrays.sort(index, (a, b) -> (a[0] - b[0]));
        int pre = 0;
        while(pre < n){
        	if(index[pre][0] < S.length() && prefixMatch(S, index[pre][0], sources[index[pre][1]])){
        		res.append(S.substring(0, index[pre][0])).append(targets[index[pre][1]]);
        		break;
        	}
        	pre++;
        }
        for(int i = pre + 1;i < n;i++){
        	int start = index[pre][0] + sources[index[pre][1]].length();
        	if(start <= index[i][0] && index[i][0] < S.length() && prefixMatch(S, index[i][0], sources[index[i][1]])){
        		res.append(S.substring(start, index[i][0])).append(targets[index[i][1]]);
        		pre = i;
        	}
        }
        if(pre < n){
        	int start = index[pre][0] + sources[index[pre][1]].length();
        	res.append(S.substring(start));
        }
        if(res.length() == 0)return S;
        return res.toString();
    }
    
    public boolean prefixMatch(String S, int start, String target){
    	for(int i = 0;i < target.length();i++){
    		if(start + i < S.length() && S.charAt(start + i) == target.charAt(i))
    			continue;
    		return false;
    	}
    	return true;
    }
}
*/

/**
 * 2018/5/17
 * 835. Image Overlap
 * 
    public int largestOverlap(int[][] A, int[][] B) {
        List<Integer> LA = new ArrayList<>();
        List<Integer> LB = new ArrayList<>();
        for(int i = 0;i < A.length;i++){
        	for(int j = 0;j < A[0].length;j++){
        		if(A[i][j] == 1)LA.add(i * 100 + j);
        		if(B[i][j] == 1)LB.add(i * 100 + j);
        	}
        }
        Map<Integer, Integer> count = new HashMap<Integer, Integer>();
        for(int i : LA){
        	for(int j : LB){
        		count.put(i - j, count.getOrDefault(i - j, 0) + 1);
        	}
        }
        int max = 0;
        for(int val : count.values()){
        	max = Math.max(max, val);
        }
        return max;
    }
    
 * 2018/5/17
 * 675. Cut Off Trees for Golf Event
 * 
class Solution {
	//TreeMap + Set -- TTL
	//row col is a small number, use PriorityQueue + boolean visit[][]
    int row;
    int col;
	public int cutOffTree(List<List<Integer>> forest) {
		row = forest.size();
		col = forest.get(0).size();
		TreeMap<Integer, Integer> map = new TreeMap<>();
		for(int i = 0;i < row;i++){
			for(int j = 0;j < col;j++){
				if(forest.get(i).get(j) > 1)
					map.put(forest.get(i).get(j), i * 100 + j);
			}
		}
		int res = 0;
		int pre = 0;
		List<Integer> indexs = new ArrayList<>();
		for(int key : map.descendingKeySet()){
			indexs.add(map.get(key));
		}
		for(int i = indexs.size() - 1;i >= 0;i--){
			int preRow = pre / 100;
			int preCol = pre % 100;
			int cur = indexs.get(i);
			int tarRow = cur / 100;
			int tarCol = cur % 100;
			int dis = distance(forest, new int[]{preRow, preCol}, new int[]{tarRow, tarCol});
			if(dis == -1)return -1;
			res += dis;
			pre = cur;
		}
        return res;
    }
	
	public int distance(List<List<Integer>> forest, int[] source, int[] target){
		int res = 1;
		Queue<int[]> queue = new LinkedList<int[]>();
		Set<String> set = new HashSet<>();
		if(source[0] == target[0] && source[1] == target[1])return 0;
		queue.add(source);
		int[][] dir = new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
		while(!queue.isEmpty()){
			int size = queue.size();
			for(int i = 0;i < size;i++){
				int[] cur = queue.poll();
				set.add(cur[0] + "\t" + cur[1]);
				for(int[] d : dir){
					int curRow = cur[0] + d[0];
					int curCol = cur[1] + d[1];
					if(curRow < row && curRow >= 0 && curCol < col && curCol >= 0 
							&& forest.get(curRow).get(curCol) != 0 && !set.contains(curRow + "\t" + curCol)){
						if(curRow == target[0] && curCol == target[1])return res;
						queue.add(new int[]{curRow, curCol});
					}
				}
			}
			res++;
		}
		return -1;
	}
}

Solution2: Accepted
class Solution {
    static int[][] dir = {{0,1}, {0, -1}, {1, 0}, {-1, 0}};
    public int cutOffTree(List<List<Integer>> forest) {
        if (forest == null || forest.size() == 0) return 0;
        int m = forest.size(), n = forest.get(0).size();
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[2] - b[2]);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (forest.get(i).get(j) > 1) {
                    pq.add(new int[] {i, j, forest.get(i).get(j)});
                }
            }
        }
        int[] start = new int[2];
        int sum = 0;
        while (!pq.isEmpty()) {
            int[] tree = pq.poll();
            int step = minStep(forest, start, tree, m, n);
            if (step < 0) return -1;
            sum += step;
            start[0] = tree[0];
            start[1] = tree[1];
        }
        return sum;
    }

    private int minStep(List<List<Integer>> forest, int[] start, int[] tree, int m, int n) {
        int step = 0;
        boolean[][] visited = new boolean[m][n];
        Queue<int[]> queue = new LinkedList<>();
        queue.add(start);
        visited[start[0]][start[1]] = true;

        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] curr = queue.poll();
                if (curr[0] == tree[0] && curr[1] == tree[1]) return step;

                for (int[] d : dir) {
                    int nr = curr[0] + d[0];
                    int nc = curr[1] + d[1];
                    if (nr < 0 || nr >= m || nc < 0 || nc >= n 
                        || forest.get(nr).get(nc) == 0 || visited[nr][nc]) continue;
                    queue.add(new int[] {nr, nc});
                    visited[nr][nc] = true;
                }
            }
            step++;
        }
        return -1;
    }
}

 * 2018/5/17
 * 493. Reverse Pairs
 * 
class Solution {//Solution1 : BinarySearchTree -- TTL
	class BSTNode{//BinarySearchTree
		int val;
		int count;
		BSTNode left;
		BSTNode right;
		
		public BSTNode(int val) {
			this.val = val;
			count = 1;
		}
	}
	
	public int search(BSTNode root, long target){
		if(root == null)return 0;
		if(root.val == target)return root.count;
		else if(root.val < target){
			return search(root.right, target);
		}else{
			return root.count + search(root.left, target);
		}
	}
	
	public BSTNode insert(BSTNode root, int val){
		if(root == null)return new BSTNode(val);
		if(root.val == val){
			root.count++;
		}else if(root.val < val){
			root.count++;
			root.right = insert(root.right, val);
		}else{
			root.left = insert(root.left, val);
		}
		return root;
	}
	
    public int reversePairs(int[] nums) {
        BSTNode root = null;
        int res = 0;
        for(int num : nums){
        	res += search(root, (long)num * 2 + 1);
        	root = insert(root, num);
        }
        return res;
    }
}

class Solution {//Solution1 : Divide and Conquer(modified merge sort) -- Accepted
	public void merge(int[] nums, int start, int mid, int end){
		int n1 = mid - start + 1;
		int n2 = end - mid;
		int[] left = new int[n1];
		int[] right = new int[n2];
		for(int i = 0;i < n1;i++){
			left[i] = nums[start + i];
		}
		for(int i = 0;i < n2;i++){
			right[i] = nums[mid + 1 + i];
		}
		int i = 0;
		int j = 0;
		for(int k = start;k <= end;k++){
			if(j >= n2 || (i < n1 && left[i] <= right[j])){
				nums[k] = left[i++];
			}else{
				nums[k] = right[j++];
			}
		}
	}
	
	public int mergeCount(int[] nums, int start, int end){
		if(start < end){
			int mid = (start + end) / 2;
			int count = mergeCount(nums, start, mid) + mergeCount(nums, mid + 1, end);
			int j = mid + 1;
			for(int i = start;i <= mid;i++){
				while(j <= end && nums[i] > (long)nums[j] * 2){
					j++;
				}
				count += j - mid - 1;
			}
			merge(nums, start, mid, end);
			return count;
		}else{
			return 0;
		}
	}
	
    public int reversePairs(int[] nums) {
        return mergeCount(nums, 0, nums.length - 1);
    }
}
*/

/**
 * 2018/5/19
 * 76. Minimum Window Substring
 * 
class Solution {
    public String minWindow(String s, String t) {
        if(s.length() < t.length() || s == null || s.length() == 0 || t == null || t.length() == 0)return "";
        Set<Character> set = new HashSet<>();//record which char should be test
        for(char c : t.toCharArray())
        	set.add(c);
        Map<Character, Integer> target = new HashMap<>();
        for(char c : t.toCharArray())
        	target.put(c, target.getOrDefault(c, 0) + 1);
        int left = 0;
        int right = 0;
        if(t.length() == 1 && s.charAt(0) == t.charAt(0))return t.charAt(0) + "";
        Map<Character, Integer> source = new HashMap<>();
        if(set.contains(s.charAt(0)))
        	source.put(s.charAt(0), 1);
        String res = s + "MakeThisStringLonger";
        while(true){
        	while(isValid(source, target)){
        		String temp = s.substring(left, right + 1);
        		res = res.length() < temp.length() ? res : temp;
        		if(set.contains(s.charAt(left))){
        			source.put(s.charAt(left), source.get(s.charAt(left)) - 1);
        		}
        		left++;
        	}
        	right++;
        	if(right >= s.length())break;
        	if(set.contains(s.charAt(right))){
        		source.put(s.charAt(right), source.getOrDefault(s.charAt(right), 0) + 1);
        	}
        }
        if(res.length() > s.length())return "";
        return res;
    }
    
    public boolean isValid(Map<Character, Integer> source, Map<Character, Integer> target){
    	for(char c : target.keySet()){
    		if(source.containsKey(c) && source.get(c) >= target.get(c))
    			continue;
    		return false;
    	}
    	return true;
    }
}

 * 2018/5/19
 * 629. K Inverse Pairs Array
 * 
	public int kInversePairs(int n, int k) {//Solution 1--TTL
        int[] pre = new int[k + 1];
        pre[0] = 1;
        for(int i = 1;i <= n;i++){
        	int[] cur = new int[k + 1];
        	for(int j = 0;j <= k;j++){
        		for(int x = Math.max(1, i - j);x <= i;x++){
        			cur[j] += pre[j - (i - x)];
        			cur[j] %= 1000000007;
        		}
        	}
        	pre = cur;
        }
        return pre[k];
    }
    
    public int kInversePairs(int n, int k) {//Solution 2--AC
        int mod = 1000000007;
        if (k > n*(n-1)/2 || k < 0) return 0;
        if (k == 0 || k == n*(n-1)/2) return 1;
        long[][] dp = new long[n+1][k+1];
        dp[2][0] = 1;
        dp[2][1] = 1;
        for (int i = 3; i <= n; i++) {
            dp[i][0] = 1;
            for (int j = 1; j <= Math.min(k, i*(i-1)/2); j++) {
                dp[i][j] = dp[i][j-1] + dp[i-1][j];
                if (j >= i) dp[i][j] -= dp[i-1][j-i];
                //above two formation derive from Solution1
                dp[i][j] = (dp[i][j]+mod) % mod;
            }
        }
        return (int) dp[n][k];
    }

 * 2018/5/19
 * 834. Sum of Distances in Tree
 * 
class Solution {
    int n;
    List<Set<Integer>> tree;
    int[] count;
    int[] res;
	
	public int[] sumOfDistancesInTree(int N, int[][] edges) {
		n = N;
		tree = new ArrayList<>();
		for(int i = 0;i < N;i++)tree.add(new HashSet<>());
		count = new int[N];
		res = new int[N];
		for(int[] edge : edges){
			tree.get(edge[0]).add(edge[1]);
			tree.get(edge[1]).add(edge[0]);
		}
		preDfs(0, new HashSet<>());
		postDfs(0, new HashSet<>());
		return res;
    }
	
	public void preDfs(int root, HashSet<Integer> seen){//update count(accurate) and res(root is accurate)
		seen.add(root);
		for(int i : tree.get(root)){
			if(!seen.contains(i)){
				preDfs(i, seen);
				count[root] += count[i];
				res[root] += res[i] + count[i];
			}
		}
		count[root]++;
	}
	
	public void postDfs(int root, HashSet<Integer> seen){//update other nodes
		seen.add(root);
		for(int i : tree.get(root)){
			if(!seen.contains(i)){
				res[i] = res[root] - count[i] + n - count[i];
				postDfs(i, seen);
			}
		}
	}
}

 * 2018/5/19
 * 335. Self Crossing
 * 
    public boolean isSelfCrossing(int[] x) {
        int len = x.length;
        if(len <= 3)return false;
        for(int i = 3;i < len;i++){
			if (x[i] >= x[i - 2] && x[i - 1] <= x[i - 3])
				return true; // Fourth line crosses first line and onward
			if (i >= 4) {
				if (x[i - 1] == x[i - 3] && x[i] + x[i - 4] >= x[i - 2])
					return true; // Fifth line meets first line and onward
			}
			if (i >= 5) {
				if (x[i - 2] - x[i - 4] >= 0 && x[i] >= x[i - 2] - x[i - 4] && x[i - 1] >= x[i - 3] - x[i - 5]
						&& x[i - 1] <= x[i - 3])
					return true; // Sixth line crosses first line and onward
			}
        }
    	return false;
    }
    
 * 2018/5/19
 * 41. First Missing Positive
 * 
class Solution {
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for(int i = 0;i < n;i++){
        	while(nums[i] > 0 && nums[i] <= n && nums[i] != nums[nums[i] - 1]){
        		swap(nums, i, nums[i] - 1);
        	}
        }
        for(int i = 0;i < n;i++){
        	if(nums[i] != i + 1){
        		return i + 1;
        	}
        }
        return n + 1;
    }
    
    public void swap(int[] nums, int i, int j){
    	int temp = nums[i];
    	nums[i] = nums[j];
    	nums[j] = temp;
    }
}

 * 2018/5/19
 * 741. Cherry Pickup
 * 
class Solution {
	int N;
	
    public int cherryPickup(int[][] grid) {
        N = grid.length;
        int[][][] dp = new int[N][N][N];
        for(int[][] r : dp){
        	for(int[] c : r){
        		Arrays.fill(c, Integer.MIN_VALUE);
        	}
        }
        return Math.max(0, DP(dp, grid, 0, 0, 0));
    }
    
    public int DP(int[][][] dp,int[][] grid, int r1,int c1, int r2){
    	int c2 = r1 + c1 - r2;
    	if(r1 == N || c1 == N || r2 == N || c2 == N || grid[r1][c1] == -1 || grid[r2][c2] == -1)
    		return Integer.MIN_VALUE;
    	else if(r1 == N - 1 && c1 == N - 1){
    		return grid[N - 1][N - 1];
    	}else if(dp[r1][c1][r2] != Integer.MIN_VALUE){
    		return dp[r1][c1][r2];
    	}else{
    		int res = grid[r1][c1];
    		if(r1 != r2)res += grid[r2][c2];
    		res += Math.max(Math.max(DP(dp, grid, r1 + 1, c1, r2 + 1), DP(dp, grid, r1, c1 + 1, r2 + 1))
    				, Math.max(DP(dp, grid, r1 + 1, c1, r2), DP(dp, grid, r1, c1 + 1, r2)));
    		dp[r1][c1][r2] = res;
    		return res;
    	}
    }
}
*/
package exercise;
