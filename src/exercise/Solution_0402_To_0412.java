/**
 * @author jhy code from 4.2 to 4.12
 * 26 questions
 */

/**
 * 2018/4/2
 * 768. Max Chunks To Make Sorted II
 * 
class Solution {
    public int maxChunksToSorted(int[] arr) {
        int res = 0;
        int index = help(arr, arr.length);
        while(index > 0){
        	res++;
        	index = help(arr, index);
        }
        return res + 1;
    }
    
    public int help(int[] arr, int end){
    	int max = -1;
    	int indexMax = 0;
    	for(int i = end - 1;i >= 0;i--){
    		if(arr[i] > max){
    			max = arr[i];
    			indexMax = i;
    		}
    	}
    	int min = Integer.MAX_VALUE;
    	for(int i = indexMax;i < end;i++){
    		if(min > arr[i])min = arr[i];
    	}
    	boolean flag = true;
    	while(indexMax >= 0 && flag){
    		int index = indexMax;
    		flag = false;
    		for(int i = 0;i < indexMax;i++){
    			if(arr[i] > min){
    				index = i;
    				flag = true;
    				break;
    			}
    		}
    		for(int i = index;i < indexMax;i++){
    			if(arr[i] < min)min = arr[i];
    		}
    		indexMax = index;
    	}
    	return indexMax;
    }
}

 * 2018/4/2
 * 65. Valid Number
 * 
    public boolean isNumber(String s) {
        s = s.trim();
        
        boolean eSeen = false;
        boolean numSeen = false;
        boolean pointSeen = false;
        
        for(int i = 0;i < s.length();i++){
        	if(s.charAt(i) >= '0' && s.charAt(i) <= '9'){
        		numSeen = true;
        	}
        	else if(s.charAt(i) == '.'){
        		if(pointSeen || eSeen){
        			return false;
        		}
        		pointSeen = true;
        	}
        	else if(s.charAt(i) == 'e'){
        		if(eSeen || !numSeen){
        			return false;
        		}
        		numSeen = false;
        		eSeen = true;
        	}
        	else if(s.charAt(i) == '-' || s.charAt(i) == '+'){
        		if(i != 0 && s.charAt(i - 1) != 'e'){
        			return false;
        		}
        	}
        	else {
				return false;
			}
        }
        return numSeen;
    }
*/

/**
 * 2018/4/3
 * 778. Swim in Rising Water
 * 
class Solution {
    public int swimInWater(int[][] grid) {
        int n = grid.length;
        int[][] max = new int[n][n];
        for(int[] line : max){
        	Arrays.fill(line, Integer.MAX_VALUE);
        }
        dfs(grid, 0, 0, max, grid[0][0]);
        return max[n - 1][n - 1];
    }
    
    int[][] dir = new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
    public boolean isValid(int i, int j, int len){
    	return i >= 0 && j >= 0 && i < len && j < len;
    }
    
    public void dfs(int[][] grid, int i, int j, int[][] max, int cur){
    	int len = grid.length;
    	if(!isValid(i, j, len) || Math.max(cur, grid[i][j]) >= max[i][j])return;
    	max[i][j] = Math.max(cur, grid[i][j]);
    	for(int[] d : dir){
    		dfs(grid, i + d[0], j + d[1], max, max[i][j]);
    	}
    }
}

 * 2018/4/3
 * 312. Burst Balloons
 * 
class Solution {
    public int maxCoins(int[] nums) {
        int[] Num = new int[nums.length + 2];
        int n = 1;
        for(int x : nums){
        	if(x > 0){
        		Num[n++] = x;
        	}
        }
        Num[0] = 1;
        Num[n++] = 1;
        
        int[][] memo = new int[n][n];
        return burst(memo, Num, 0, n - 1);
    }
    
    public int burst(int[][] memo, int[] nums, int left, int right){
    	if(left + 1 == right){
    		return 0;
    	}
    	if(memo[left][right] > 0)return memo[left][right];
    	int res = 0;
    	for(int i = left + 1;i < right;i++){
    		res = Math.max(res, nums[i] * nums[left] * nums[right] + burst(memo, nums, left, i) + burst(memo, nums, i, right));
    	}
    	memo[left][right] = res;
    	return res;
    }
}
*/

/**
 * 2018/4/4
 * 632. Smallest Range
 * 
    public int[] smallestRange(List<List<Integer>> nums) {
        int[] next = new int[nums.size()];
        boolean flag = true;
        int resx = 0;
        int resy = Integer.MAX_VALUE;
        int max = 0;
        PriorityQueue<Integer> min_queue = new PriorityQueue<>((i,j) -> (nums.get(i).get(next[i]) - nums.get(j).get(next[j])));
        for(int i = 0;i < nums.size();i++){
        	min_queue.add(i);
        	max = Math.max(max, nums.get(i).get(0));
        }
        for(int i = 0;i < nums.size() && flag;i++){
        	for(int j = 0;j < nums.get(i).size();j++){
        		int minIndex = min_queue.poll();
        		if(resy - resx > max - nums.get(minIndex).get(next[minIndex])){
        			resx = nums.get(minIndex).get(next[minIndex]);
        			resy = max;
        		}
        		next[minIndex]++;
        		if(next[minIndex] == nums.get(minIndex).size()){
        			flag = false;
        			break;
        		}
        		min_queue.add(minIndex);
        		max = Math.max(max, nums.get(minIndex).get(next[minIndex]));
        	}
        }
        return new int[]{resx, resy};
    }
    
 * 2018/4/4
 * 689. Maximum Sum of 3 Non-Overlapping Subarrays
 * 
    public int[] maxSumOfThreeSubarrays(int[] nums, int k) {
        int n = nums.length;
        int maxSum = 0;
    	int[] sum = new int[n + 1];
        int[] posLeft = new int[n];
        int[] posRight = new int[n];
        int[] res = new int[3];
        for(int i = 0;i < n;i++)sum[i + 1] = sum[i] + nums[i];
        //DP for start index of posLeft
        for(int i = k, tot = sum[k] - sum[0];i < n;i++){
        	if(sum[i + 1] - sum[i + 1 - k] > tot){
        		posLeft[i] = i + 1 - k;
        		tot = sum[i + 1] - sum[i + 1 - k];
        	}
        	else posLeft[i] = posLeft[i - 1];
        }
        //DP for start index of posRight
        posRight[n - k] = n - k;
        for(int i = n - k - 1, tot = sum[n] - sum[n - k];i >= 0;i--){
        	if(sum[i + k] - sum[i] >= tot){
        		posRight[i] = i;
        		tot = sum[i + k] - sum[i];
        	}
        	else posRight[i] = posRight[i + 1];
        }
        for(int i = k;i <= n - 2 * k;i++){
        	int l = posLeft[i - 1];
        	int r = posRight[i + k];
        	int tot = (sum[i + k] - sum[i]) + (sum[l + k] - sum[l]) + (sum[r + k] - sum[r]);
        	if(tot > maxSum){
        		maxSum = tot;
        		res[0] = l;
        		res[1] = i;
        		res[2] = r;
        	}
        }
        return res;
    }
*/

/**
 * 2018/4/5
 * 761. Special Binary String
 * 
	public String makeLargestSpecial(String S) {
		int count = 0;
		int i = 0;
		List<String> res = new ArrayList<>();
		for(int j = 0;j < S.length();j++){
			if(S.charAt(j) == '1')count++;
			else count--;
			if(count == 0){
				res.add('1' + makeLargestSpecial(S.substring(i + 1,j)) + '0');
				i = j + 1;
			}
		}
		Collections.sort(res, Collections.reverseOrder());
		return String.join("", res);
	}
*/

/**
 * 2018/4/6
 * 410. Split Array Largest Sum
 * 
class Solution {
	int[] sum;//[i]为加到nums 0 -> i - 1的值
	int[][] dp;
	int n;
	
    public int splitArray(int[] nums, int m) {
    	if(nums == null || nums.length == 0)return 0;
        if(m == nums.length){
        	int max = Integer.MIN_VALUE;
        	for(int num : nums)max = Math.max(max, num);
        	return max;
        }
        n = nums.length;
        sum = new int[n + 1];
        sum[0] = nums[0];
        for(int i = 1;i < n + 1;i++){
        	sum[i] = sum[i - 1] + nums[i - 1];
        }
        dp = new int[n][m + 1];
        for(int i = 0;i < n;i++){
        	Arrays.fill(dp[i], -1);
        }
        return help(nums, 0, m);
    }
    
    public int help(int[] nums, int start, int m){
    	if(m == 1)return sum[n] - sum[start];
    	if(start >= n || m <= 0)return 0;
    	if(dp[start][m] != -1)return dp[start][m];
    	if(n - start <= m){
    		int max = Integer.MIN_VALUE;
    		for(int i = start;i < n;i++){
    			max = Math.max(max, nums[i]);
    		}
    		dp[start][m] = max;
    	}
    	else{
    		int min = Integer.MAX_VALUE;
    		for(int i = start;i <= n - m;i++){
    			int temp = sum[i + 1] - sum[start];
    			temp = Math.max(temp, help(nums, i + 1, m - 1));
    			min = Math.min(min, temp);
    		}
    		dp[start][m] = min;
    	}
    	return dp[start][m];
    }
}

 * 2018/4/6
 * 668. Kth Smallest Number in Multiplication Table
 * 
class Solution {
    public int findKthNumber(int m, int n, int k) {
        int low = 1;
        int high = m * n;
        while(low < high){
        	int mid = (low + high) / 2;
        	if(enough(mid, m, n, k)){
        		high = mid;
        	}
        	else{
        		low = mid + 1;
        	}
        }
        return low;
    }
    
    public boolean enough(int x, int m, int n, int k){
    	int count = 0;
    	for(int i = 1;i <= m;i++){
    		count += Math.min(n, x / i);
    	}
    	return count >= k;
    }
}
*/

/**
 * 2018/4/7
 * 154. Find Minimum in Rotated Sorted Array II
 * 
    public int findMin(int[] nums) {
        int low = 0;
        int high = nums.length - 1;
        while(low < high){
        	int mid = (low + high) / 2;
        	if(nums[mid] > nums[high]){
        		low = mid + 1;
        	}
        	else if(nums[mid] < nums[high]){
        		high = mid;
        	}
        	else{
        		if(nums[low] == nums[mid]){
        			low++;
        			high--;
        		}
        		else{
        			high = mid;
        		}
        	}
        }
        return nums[low];
    }

 * 2018/4/7
 * 679. 24 Game
 * 
class Solution {
	boolean res = false;
	final double eps = 0.001;
	
    public boolean judgePoint24(int[] nums) {
        List<Double> arr = new ArrayList<>();
        for(int num : nums)arr.add((double)num);
        help(arr);
        return res;
    }
    
    public void help(List<Double> arr){
    	if(res)return;
    	if(arr.size() == 1){
    		if(Math.abs(arr.get(0) - 24.0) < eps){
    			res = true;
    		}
    		return;
    	}
    	for(int i = 0;i < arr.size();i++){
    		for(int j = 0;j < i;j++){
    			List<Double> next = new ArrayList<>();
				Double p1 = arr.get(i), p2 = arr.get(j);
				next.addAll(Arrays.asList(p1 + p2, p1 - p2, p2 - p1, p1 * p2));
				if (Math.abs(p2) > eps)next.add(p1 / p2);
				if (Math.abs(p1) > eps)next.add(p2 / p1);
				
				arr.remove(i);
				arr.remove(j);
				for(double n : next){
					arr.add(n);
					help(arr);
					arr.remove(arr.size() - 1);
				}
				arr.add(j, p2);
				arr.add(i, p1);
    		}
    	}
    }
}
*/

/**
 * 2018/4/8
 * 753. Cracking the Safe
 * 
class Solution {
    Set<String> seen;
    StringBuilder res;
	
	public String crackSafe(int n, int k) {
		if(n == 1 && k == 1)return "0";
		seen = new HashSet<>();
    	res = new StringBuilder();
    	StringBuilder sb = new StringBuilder();
    	for(int i = 0;i < n - 1;i++){
    		sb.append("0");
    	}
    	dfs(sb.toString(), k);
    	res.append(sb.toString());
    	return res.toString();
    }
	
	public void dfs(String str, int k){
		for(int x = 0;x < k;x++){
			String temp = str + x;
			if(!seen.contains(temp)){
				seen.add(temp);
				dfs(temp.substring(1), k);
				res.append(x);
			}
		}
	}
}

 * 2018/4/8
 * 815. Bus Routes
 * 
    public int numBusesToDestination(int[][] routes, int S, int T) {
    	if(S == T) return 0;
        Map<Integer, Set<Integer>> stepBus = new HashMap<>();
        for(int i = 0;i < routes.length;i++){
        	for(int step : routes[i]){
        		if(stepBus.containsKey(step)){
        			stepBus.get(step).add(i);
        		}
        		else{
        			Set<Integer> temp = new HashSet<>();
        			temp.add(i);
        			stepBus.put(step, temp);
        		}
        	}
        }
        boolean[] visit = new boolean[routes.length];
        Set<Integer> pass = new HashSet<>();//已经经过的站点
        List<Integer> step = new ArrayList<>();
        step.add(S);
        pass.add(S);
        int res = 1;
        while(!step.isEmpty()){
        	List<Integer> temp = new ArrayList<>();
        	for(int st : step){
        		for(int bus : stepBus.get(st)){
        			if(!visit[bus]){
        				for(int num : routes[bus]){
        					if(!pass.contains(num)){
        						if(num == T)return res;
        						else{
        							pass.add(num);
        							temp.add(num);
        						}
        					}
        				}
        				visit[bus] = true;
        			}
        		}
        	}
        	res++;
        	step = temp;
        }
        return -1;
    }
    
 * 2018/4/8
 * 813. Largest Sum of Averages
 * 
class Solution {
	double[] sum;//[i]为加到nums 0 -> i - 1的值
	double[][] dp;
	int n;
	
    public double largestSumOfAverages(int[] A, int K) {
    	if(A == null || A.length == 0)return 0;
        n = A.length;
        sum = new double[n + 1];
        sum[0] = 0;
        for(int i = 1;i < n + 1;i++){
        	sum[i] = sum[i - 1] + A[i - 1];
        }
        if(K == A.length){
        	return sum[n];
        }
        dp = new double[n][K + 1];
        for(int i = 0;i < n;i++){
        	Arrays.fill(dp[i], -1);
        }
        return help(A, 0, K);
    }
    
    public double help(int[] nums, int start, int m){
    	if(m == 1)return (sum[n] - sum[start]) / (n - start);
    	if(start >= n || m <= 0)return 0;
    	if(dp[start][m] != -1)return dp[start][m];
    	if(n - start <= m){
    		double max = sum[n] - sum[start];
    		dp[start][m] = max;
    	}
    	else{
    		double max = 0;
    		for(int i = start;i <= n - m;i++){
    			double tempAvg = (sum[i + 1] - sum[start]) / (i - start + 1);
    			max = Math.max(max, tempAvg + help(nums, i + 1, m - 1));
    		}
    		dp[start][m] = max;
    	}
    	return dp[start][m];
    }
}

 * 2018/4/8
 * 814. Binary Tree Pruning
 * 
class Solution {
    public TreeNode pruneTree(TreeNode root) {
        if(isAllZero(root))return null;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
        	int size = queue.size();
        	for(int i = 0;i < size;i++){
        		TreeNode temp = queue.poll();
        		if(isAllZero(temp.left))temp.left = null;
        		else queue.add(temp.left);
	        	if(isAllZero(temp.right))temp.right = null;
	        	else queue.add(temp.right);
        	}
        }
        return root;
    }
    
    public boolean isAllZero(TreeNode root){
    	if(root == null)return true;
    	if(root.val == 1)return false;
    	return isAllZero(root.left) && isAllZero(root.right);
    }
}

 * 2018/4/8
 * 812. Largest Triangle Area
 * 
class Solution {
    public double largestTriangleArea(int[][] points) {
        double max = -1;
        for(int i = 0;i < points.length - 2;i++){
        	for(int j = i + 1;j < points.length - 1;j++){
        		for(int k = j + 1;k < points.length;k++){
        			max = Math.max(max, area(points[i], points[j], points[k]));
        		}
        	}
        }
    	return max;
    }
    
    public double area(int[] pointA, int[] pointB, int[] pointC){
    	double a = Math.sqrt(Math.pow(pointA[0] - pointB[0], 2) + Math.pow(pointA[1] - pointB[1], 2));
    	double b = Math.sqrt(Math.pow(pointA[0] - pointC[0], 2) + Math.pow(pointA[1] - pointC[1], 2));
    	double c = Math.sqrt(Math.pow(pointC[0] - pointB[0], 2) + Math.pow(pointC[1] - pointB[1], 2));
    	if(a + b > c && b + c > a && a + c > b){
        	double p = (a + b + c) / 2;
        	double S = Math.sqrt(p * (p - a) * (p - b) * (p - c));
        	return S;
    	}
    	return 0;
    }
}
*/

/**
 * 2018/4/9
 * 514. Freedom Trail
 * 
class Solution {//dp and dfs
	Map<String, Integer> map;
	
    public int findRotateSteps(String ring, String key) {
    	map = new HashMap<>();
        return help(ring, 0, key, 0);
    }
    
    public int help(String ring, int point, String key, int index){
    	if(index == key.length())return 0;
    	int min = Integer.MAX_VALUE;
    	for(int i = 0;i < ring.length();i++){
    		if(ring.charAt(i) == key.charAt(index)){
    			int temp = Math.abs(i - point);
    			int dis = Math.min(temp, ring.length() - temp) + 1;//1为需要按按钮
    			if(map.containsKey(i + "conject" + (index + 1))){
    				min = Math.min(min, dis + map.get(i + "conject" + (index + 1)));
    			}
    			else{
    				min = Math.min(min, dis + help(ring, i, key, index + 1));
    			}
    		}
    	}
    	map.put(point + "conject" + index, min);
    	return min;
    }
}

 * 2018/4/9
 * 749. Contain Virus
 * 
class Solution {
	int R,C;
	int[] dr = new int[]{0,0,1,-1};
	int[] dc = new int[]{1,-1,0,0};
	Set<Integer> seen;
	List<Set<Integer>> frontier;
	List<Set<Integer>> region;
	List<Integer> perimeter;
	
    public int containVirus(int[][] grid) {
        R = grid.length;
        C = grid[0].length;
        int res = 0;
        while(true){
        	seen = new HashSet<>();
        	frontier = new ArrayList<>();
        	region = new ArrayList<>();
        	perimeter = new ArrayList<>();
        	
        	for(int i = 0;i < R;i++){
        		for(int j = 0;j < C;j++){
        			if(grid[i][j] == 1 && !seen.contains(i * C + j)){
        				region.add(new HashSet<>());
        				frontier.add(new HashSet<>());
        				perimeter.add(0);
        				dfs(grid, i, j);
        			}
        		}
        	}
        	
        	if(region.isEmpty())break;
        	int triageIndex = 0;
        	for(int i = 0;i < frontier.size();i++){
        		if(frontier.get(i).size() > frontier.get(triageIndex).size()){
        			triageIndex = i;
        		}
        	}
        	
        	res += perimeter.get(triageIndex);
        	for(int i = 0;i < region.size();i++){
        		if(i == triageIndex){
        			for(int code : region.get(i)){
        				grid[code / C][code % C] = -1;
        			}
        		}
        		else{
        			for(int code : region.get(i)){
        				int r = code / C, c = code % C;
        				for(int j =  0;j < 4;j++){
        					int nr = r + dr[j];
        					int nc = c + dc[j];
        					if (nr >= 0 && nr < R && nc >= 0 && nc < C && grid[nr][nc] == 0){
        						grid[nr][nc] = 1;
        					}
        				}
        			}
        		}
        	}
        }
        return res;
    }
    
    public void dfs(int[][] grid, int r, int c){
    	if(!seen.contains(r * C + c)){
    		seen.add(r * C + c);
    		int N = region.size();
    		region.get(N - 1).add(r * C + c);
    		for(int i = 0;i < 4;i++){
    			int nr = r + dr[i];
    			int nc = c + dc[i];
				if (nr >= 0 && nr < R && nc >= 0 && nc < C) {
					if (grid[nr][nc] == 1) {
						dfs(grid, nr, nc);
					} else if(grid[nr][nc] == 0){
						frontier.get(N - 1).add(nr * C + nc);
						perimeter.set(N - 1, perimeter.get(N - 1) + 1);
					}
				}
    		}
    	}
    }
}

 * 2018/4/9
 * 128. Longest Consecutive Sequence
 * 
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for(int num : nums)set.add(num);
        int res = 0;
        for(int num : nums){
        	if(!set.contains(num - 1)){
        		int tempNum = num;
        		int count = 1;
        		while(set.contains(tempNum + 1)){
        			tempNum++;
        			count++;
        		}
        		res = Math.max(res, count);
        	}
        }
        return res;
    }
*/

/**
 * 2018/4/10
 * 42. Trapping Rain Water
 * 
    public int trap(int[] height) {
        if(height == null || height.length == 0)return 0;
    	int res = 0;
        int cur = 0;
        Stack<Integer> stack = new Stack<>();
        while(cur < height.length){
        	while(!stack.isEmpty() && height[cur] > height[stack.peek()]){
        		int top = stack.pop();//底部高度
        		if(stack.isEmpty()) break;
        		int index = stack.peek();//左侧高度,cur为右侧高度
        		int dis = cur - index - 1;//左右两侧距离
        		int hei = Math.min(height[index], height[cur]) - height[top];
        		res += dis * hei;
        	}
        	stack.push(cur++);
        }
        return res;
    }
    
 * 2018/4/10
 * 407. Trapping Rain Water II
 * 
class Solution {
	class Cell{
		int row;
		int col;
		int height;
		public Cell(int row, int col, int height){
			this.row = row;
			this.col = col;
			this.height = height;
		}
	}
	
    public int trapRainWater(int[][] heightMap) {
        if(heightMap == null || heightMap.length == 0 || heightMap[0] == null || heightMap[0].length == 0)return 0;
        PriorityQueue<Cell> queue = new PriorityQueue<>(1, new Comparator<Cell>() {
			public int compare(Cell o1, Cell o2) {
				return o1.height - o2.height;
			}
		});
        int res = 0;
        //initial queue
        int R = heightMap.length;
        int C = heightMap[0].length;
        boolean[][] visit = new boolean[R][C];
        for(int i = 0;i < R;i++){
        	visit[i][0] = true;
        	visit[i][C - 1] = true;
        	queue.add(new Cell(i, 0, heightMap[i][0]));
        	queue.add(new Cell(i, C - 1, heightMap[i][C - 1]));
        }
        for(int i = 0;i < C;i++){
        	visit[0][i] = true;
        	visit[R - 1][i] = true;
        	queue.add(new Cell(0, i, heightMap[0][i]));
        	queue.add(new Cell(R - 1, i, heightMap[R - 1][i]));
        }
        //element of queue can be assumed as the frontier
        int[][] dir = new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
        while(!queue.isEmpty()){
        	Cell temp = queue.poll();
        	for(int[] d : dir){
        		int row = temp.row + d[0];
        		int col = temp.col + d[1];
        		if(row >= 0 && col >= 0 && row < R && col < C && !visit[row][col]){
        			visit[row][col] = true;
        			res += Math.max(0, temp.height - heightMap[row][col]);
        			queue.add(new Cell(row, col, Math.max(temp.height, heightMap[row][col])));
        		}
        	}
        }
        return res;
    }
}

 * 2018/4/10
 * 329. Longest Increasing Path in a Matrix
 * 
class Solution {
	int[][] dp;
	int row;
	int col;
	int[][] dir = new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
	
    public int longestIncreasingPath(int[][] matrix) {
        if(matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)return 0;
        row = matrix.length;
        col = matrix[0].length;
        dp = new int[row][col];
        int max = 0;
        for(int i = 0;i < row;i++){
        	for(int j = 0;j < col;j++){
        		max = Math.max(max, help(matrix, i, j));
        	}
        }
        return max;
    }
    
    public int help(int[][] matrix, int i, int j){
    	if(dp[i][j] != 0)return dp[i][j];
    	int max = 0;
    	for(int[] d : dir){
    		int r = i + d[0];
    		int c = j + d[1];
    		if(r >= 0 && c >= 0 && r < row && c < col && matrix[r][c] < matrix[i][j]){
    			max = Math.max(max, help(matrix, r, c));
    		}
    	}
    	dp[i][j] = max + 1;
    	return dp[i][j];
    }
}

 * 2018/4/10
 * 502. IPO
 * 
    public int findMaximizedCapital(int k, int W, int[] Profits, int[] Capital) {
        PriorityQueue<int[]> bigger = new PriorityQueue<>((a, b) -> (a[0] - b[0]));//0-> capital, 1-> profit
        PriorityQueue<int[]> able = new PriorityQueue<>((a, b) -> (a[1] - b[1]));//可获取的值
        for(int i = 0;i < Profits.length;i++){
        	bigger.add(new int[]{Capital[i], Profits[i]});
        }
        while(k > 0){
        	while(!bigger.isEmpty() && bigger.peek()[0] <= W){
        		able.add(bigger.poll());
        	}
        	if(able.isEmpty())break;
        	W += able.poll()[1];
        	k--;
        }
        return W;
    }
*/

/**
 * 2018/4/11
 * 517. Super Washing Machines
 * 
    public int findMinMoves(int[] machines) {
        if(machines == null || machines.length == 0)return 0;
        int sum = 0;
        for(int num : machines)sum += num;
        if(sum == 0)return 0;
        if(sum % machines.length != 0)return -1;
        int target = sum / machines.length;
        int max = 0;
        int count = 0;
        for(int num : machines){
        	count += target - num;
        	max = Math.max(max, Math.max(Math.abs(count), num - target));
        }
        return max;
    }
    
 * 2018/4/11
 * 488. Zuma Game
 * 
class Solution {
    public int findMinStep(String board, String hand) {
        int[] ch = new int[26];
        for(int c : hand.toCharArray()){
        	ch[c - 'A']++;
        }
        return help(board, ch);
    }
    
    public int help(String board, int[] hand){
    	if("".equals(board))return 0;
    	int res = 2 * board.length() + 1;
    	int i = 0;
    	while(i < board.length()){
    		int j = i++;
    		while(i < board.length() && board.charAt(i) == board.charAt(j))i++;
    		int need = Math.max(0, 3 - (i - j));//需要的数量
    		if(need <= hand[board.charAt(j) - 'A']){
    			hand[board.charAt(j) - 'A'] -= need;
    			int temp = help(board.substring(0, j) + board.substring(i), hand);
    			if(temp >= 0){
    				res = Math.min(res, need + temp);
    			}
    			hand[board.charAt(j) - 'A'] += need;
    		}
    	}
    	return res == 2 * board.length() + 1 ? -1 : res;
    }
}
*/

/**
 * 2018/4/12
 * 768. Max Chunks To Make Sorted II
 * 
class Solution {
    public List<String> basicCalculatorIV(String expression, String[] evalVars, int[] evalInts) {
        Map<String, Integer> evalMap = new HashMap();
        for (int i = 0; i < evalVars.length; ++i)
            evalMap.put(evalVars[i], evalInts[i]);

        return parse(expression).evaluate(evalMap).toList();
    }

    public Poly make(String expr) {
        Poly ans = new Poly();
        List<String> list = new ArrayList();
        if (Character.isDigit(expr.charAt(0))) {
            ans.update(list, Integer.valueOf(expr));
        } else {
            list.add(expr);
            ans.update(list, 1);
        }
        return ans;
    }

    public Poly combine(Poly left, Poly right, char symbol) {
        if (symbol == '+') return left.add(right);
        if (symbol == '-') return left.sub(right);
        if (symbol == '*') return left.mul(right);
        throw null;
    }

    public Poly parse(String expr) {
        List<Poly> bucket = new ArrayList();
        List<Character> symbols = new ArrayList();
        int i = 0;
        while (i < expr.length()) {
            if (expr.charAt(i) == '(') {
                int bal = 0, j = i;
                for (; j < expr.length(); ++j) {
                    if (expr.charAt(j) == '(') bal++;
                    if (expr.charAt(j) == ')') bal--;
                    if (bal == 0) break;
                }
                bucket.add(parse(expr.substring(i+1, j)));
                i = j;
            } else if (Character.isLetterOrDigit(expr.charAt(i))) {
                int j = i;
                search : {
                    for (; j < expr.length(); ++j)
                        if (expr.charAt(j) == ' ') {
                            bucket.add(make(expr.substring(i, j)));
                            break search;
                        }
                    bucket.add(make(expr.substring(i)));
                }
                i = j;
            } else if (expr.charAt(i) != ' ') {
                symbols.add(expr.charAt(i));
            }
            i++;
        }

        for (int j = symbols.size() - 1; j >= 0; --j)
            if (symbols.get(j) == '*')
                bucket.set(j, combine(bucket.get(j), bucket.remove(j+1), symbols.remove(j)));

        if (bucket.isEmpty()) return new Poly();
        Poly ans = bucket.get(0);
        for (int j = 0; j < symbols.size(); ++j)
            ans = combine(ans, bucket.get(j+1), symbols.get(j));

        return ans;
    }
}

class Poly {
    HashMap<List<String>, Integer> count;
    Poly() {count = new HashMap();}

    void update(List<String> key, int val) {
        this.count.put(key, this.count.getOrDefault(key, 0) + val);
    }

    Poly add(Poly that) {
        Poly ans = new Poly();
        for (List<String> k: this.count.keySet())
            ans.update(k, this.count.get(k));
        for (List<String> k: that.count.keySet())
            ans.update(k, that.count.get(k));
        return ans;
    }

    Poly sub(Poly that) {
        Poly ans = new Poly();
        for (List<String> k: this.count.keySet())
            ans.update(k, this.count.get(k));
        for (List<String> k: that.count.keySet())
            ans.update(k, -that.count.get(k));
        return ans;
    }

    Poly mul(Poly that) {
        Poly ans = new Poly();
        for (List<String> k1: this.count.keySet())
            for (List<String> k2: that.count.keySet()) {
                List<String> kNew = new ArrayList();
                for (String x: k1) kNew.add(x);
                for (String x: k2) kNew.add(x);
                Collections.sort(kNew);
                ans.update(kNew, this.count.get(k1) * that.count.get(k2));
            }
        return ans;
    }

    Poly evaluate(Map<String, Integer> evalMap) {
        Poly ans = new Poly();
        for (List<String> k: this.count.keySet()) {
            int c = this.count.get(k);
            List<String> free = new ArrayList();
            for (String token: k) {
                if (evalMap.containsKey(token))
                    c *= evalMap.get(token);
                else
                    free.add(token);
            }
            ans.update(free, c);
        }
        return ans;
    }

    int compareList(List<String> A, List<String> B) {
        int i = 0;
        for (String x: A) {
            String y = B.get(i++);
            if (x.compareTo(y) != 0) return x.compareTo(y);
        }
        return 0;
    }
    List<String> toList() {
        List<String> ans = new ArrayList();
        List<List<String>> keys = new ArrayList(this.count.keySet());
        Collections.sort(keys, (a, b) ->
            a.size() != b.size() ? b.size() - a.size() : compareList(a, b));

        for (List<String> key: keys) {
            int v = this.count.get(key);
            if (v == 0) continue;
            StringBuilder word = new StringBuilder();
            word.append("" + v);
            for (String token: key) {
                word.append('*');
                word.append(token);
            }
            ans.add(word.toString());
        }
        return ans;
    }
}
*/

package exercise;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

