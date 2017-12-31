/**
 * @author jhy
 * code from 7.26 to 8.2
 * 33 questions
 */


/**
 * 2017/7/26
 * 442. Find All Duplicates in an Array
 * 
    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> result = new ArrayList<Integer>();
        for(int i = 0;i < nums.length;i++){
        	int index = Math.abs(nums[i]) - 1;
        	if(nums[index] < 0){
        		result.add(Math.abs(nums[i]));
        	}
        	nums[index] = -nums[index];
        }
        return result;
    }
    
    
 * 2017/7/26
 * 406. Queue Reconstruction by Height
 * 
	public int[][] reconstructQueue(int[][] people) {
		Arrays.sort(people, new Comparator<int[]>() {
			public int compare(int[] a, int[] b) {
				if (b[0] == a[0])
					return a[1] - b[1];
				return b[0] - a[0];
			}
		});
		int i = 0;
		while(i < people.length && people[i][0] == people[0][0]){
			i++;
		}
		for(;i < people.length;i++){
			insert(people,i);
		}
		return people;
	}

	public void insert(int[][] people ,int i){
		int index = people[i][1];
		int[] temp = people[i];
		for(int j = i;j > index;j--){
			people[j] = people[j - 1];
		}
		people[index] = temp;
	}
	
	
 * 2017/7/26
 * 515. Find Largest Value in Each Tree Row
 * 
    public List<Integer> largestValues(TreeNode root) {
        List<Integer> result = new LinkedList<Integer>();
        if(root == null)return result;
        Queue<TreeNode> que = new LinkedList<TreeNode>();
        que.add(root);
        while(!que.isEmpty()){
        	int size = que.size();
        	int max = que.peek().val;
        	for(int i = 0;i < size;i++){
        		root = que.poll();
        		if(root.val > max)max = root.val;
        		if(root.left != null)que.add(root.left);
        		if(root.right != null)que.add(root.right);
        	}
        	result.add(max);
        }
        return result;
    }
    
    
 * 2017/7/26
 * 413. Arithmetic Slices
 * 
public class Solution {
    public int numberOfArithmeticSlices(int[] A) {
        if(A.length < 3)return 0;
        int sum = 0;
        int j = 1;
        int i = 0;
        while(j != -1){
        	j = indexOfArithmeticSlices(A,i);//从i开始算，到j位置满足要求
        	sum += numOfArray(j - i + 1);//满足的个数
        	i = j;
        }
        return sum;
    }
    
    public int indexOfArithmeticSlices(int[] A,int i){
    	if(i > A.length - 3)return -1;
    	int temp = A[i + 1] - A[i];
    	for(int k = i + 1;k < A.length - 1;k++){
    		if(A[k + 1] - A[k] != temp)return k;
    	}
    	return A.length - 1;
    }
    
    public int numOfArray(int i){
    	if(i < 3)return 0;
    	return (i - 2) * (i - 1) / 2;
    }
}


 * 2017/7/26
 * 540. Single Element in a Sorted Array
 * 
    public int singleNonDuplicate(int[] nums) {
        int left = 0;
        int right = nums.length;
        while(left < right){
        	int mid = left + (right - left) / 2;
        	if(mid == 0 && nums[mid] != nums[mid + 1]){
        		return nums[mid];
        	}
        	else if(mid == nums.length - 1 && nums[mid] != nums[mid - 1]){
        		return nums[mid];
        	}
			if (nums[mid] == nums[mid - 1]) {
				if (mid % 2 == 0)
					right = mid - 2;
				else
					left = mid + 1;
			}
        	else if(nums[mid] == nums[mid + 1]){
				if (mid % 2 == 0)
					left = mid + 2;
				else
					right = mid - 1;
        	}
        	else return nums[mid];
        }
        return nums[left];
    }
*/

/**
 * 2017/7/27
 * 526. Beautiful Arrangement
 * 
public class Solution {
	public int count = 0;
    public int countArrangement(int N) {
    	int[] arr = new int[N + 1];
    	for(int i = 0;i <= N;i++){
    		arr[i] = i;
    	} 
		for(int j = N;j > 0;j--){
			swap(arr,j,N);
			countArray(arr,N);
			swap(arr,j,N);
		}
    	return count;
    }
    public void countArray(int[] arr,int n){
    	if(n == 0){
    		count++;
    		return;
    	}
    	if(arr[n] % n == 0 || n % arr[n] == 0){
			countArray(arr,n - 1);
    		for(int j = n - 2;j > 0;j--){
    			swap(arr,j,n-1);
    			countArray(arr,n - 1);
    			swap(arr,j,n-1);
    		}
    	}
    	
    }
    public void swap(int[] A,int i,int j){
    	int temp = A[i];
    	A[i] = A[j];
    	A[j] = temp;
    }
}


 * 2017/7/27
 * 508. Most Frequent Subtree Sum
 * 
public class Solution {
	Map<Integer,Integer> map = new HashMap<Integer,Integer>();
    public int[] findFrequentTreeSum(TreeNode root) {
    	if(root == null){
    		int[] result = new int[0];
    		return result;
    	}
        findTreeSum(root);
        List<Integer> tempResult = new ArrayList<Integer>();
        Set<Integer> keys = map.keySet();
        int max = Integer.MIN_VALUE;
		for (int key : keys) {
        	if(map.get(key) > max){
        		max = map.get(key);
        	}
        }
        for(int key:keys){
        	if(map.get(key) == max){
        		tempResult.add(key);
        	}
        }
        int[] result = new int[tempResult.size()];
        for(int i = 0;i < result.length;i++){
        	result[i] = tempResult.get(i);
        }
        return result;
    }
	
    public int findTreeSum(TreeNode root){
    	int sum = 0;
    	if(root.left == null && root.right == null){
    		map.put(root.val, map.getOrDefault(root.val, 0) + 1);
    		return root.val;
    	}
    	else if(root.left != null && root.right == null){
    		sum = root.val + findTreeSum(root.left);
    	}
    	else if(root.left == null && root.right != null){
    		sum = root.val + findTreeSum(root.right);
    	}
    	else{
    		sum = root.val + findTreeSum(root.right) + findTreeSum(root.left);
    	}
		map.put(sum, map.getOrDefault(sum, 0) + 1);
		return sum;
    }
}


 * 2017/7/27
 * 495. Teemo Attacking
 * 
    public int findPoisonedDuration(int[] timeSeries, int duration) {
    	if(timeSeries.length == 0)return 0;
        int sum = 0;
        Arrays.sort(timeSeries);
        sum += duration;
        for(int i = 1;i < timeSeries.length;i++){
        	if(timeSeries[i] < timeSeries[i - 1] + duration){
        		sum += (timeSeries[i] - timeSeries[i - 1]);
        	}
        	else{
        		sum += duration;
        	}
        }
        return sum;
    }
    
    
 * 2017/7/27
 * 462. Minimum Moves to Equal Array Elements II
 * 
    public int minMoves2(int[] nums) {
        Arrays.sort(nums);
        int sum = 0;
        int left = 0;
        int right = nums.length - 1;
        while(left < right){
        	sum += nums[right] - nums[left];
        	left++;
        	right--;
        }
        return sum;
    }
 */

/**
 * 2017/7/28
 * 260. Single Number III
 * 
    public int[] singleNumber(int[] nums) {
        int index = 0;
        for(int i = 0;i < nums.length;i++){
        	index ^= nums[i];
        }
        index &= -index;
        int[] result = new int[]{0,0};
        for(int num: nums){
        	if((num & index) == 0){
        		result[0] ^= num;
        	}
        	else{
        		result[1] ^= num;
        	}
        }
        return result;
    }
 */

/**
 * 2017/7/29
 * 451. Sort Characters By Frequency
 * 
    public String frequencySort(String s) {
    	StringBuffer result = new StringBuffer();
    	if(s.length() == 0)return result.toString();
        Map<Character, Integer> map = new HashMap<Character, Integer>();  
        for(int i = 0;i < s.length();i++){
        	map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0) + 1);
        }
        //将map.entrySet()转换成list  
        List<Map.Entry<Character, Integer>> list = new ArrayList<Map.Entry<Character, Integer>>(map.entrySet());  
        Collections.sort(list, new Comparator<Map.Entry<Character, Integer>>() {  
            //降序排序  
            public int compare(Map.Entry<Character, Integer> o1, Map.Entry<Character, Integer> o2) {  
                return o2.getValue() - o1.getValue();  
            }  
        });  
  
        for (Map.Entry<Character, Integer> mapping : list) {  
        	for(int i = 0;i < mapping.getValue();i++){
        		result.append(mapping.getKey());
        	}
        }  
        return result.toString();
    }
    
    
 * 2017/7/29
 * 609. Find Duplicate File in System
 * 
    public List<List<String>> findDuplicate(String[] paths) {
    	List<List<String>> result = new ArrayList<List<String>>();
    	if(paths.length == 0)return result;
    	Map<String,Set<String>> map = new HashMap<String,Set<String>>();
    	for(String path : paths){
    		String[] temp = path.split("\\s+");
    		for(int i = 1;i < temp.length;i++){
    			int index = temp[i].indexOf("(");
    			String content = temp[i].substring(index);
    			String filename = temp[0] + "/" + temp[i].substring(0, index);
    			Set<String> filenames = map.getOrDefault(content, new HashSet<String>());
    			filenames.add(filename);
    			map.put(content, filenames);
    		}
    	}
		for (String key : map.keySet()) {
			if (map.get(key).size() > 1) {
				result.add(new ArrayList<String>(map.get(key)));
			}
		}
		return result;
	}
	
	
 * 2017/7/29
 * 553. Optimal Division
 * 
    public String optimalDivision(int[] nums) {
    	if(nums.length == 0)return null;
        if(nums.length == 1)return nums[0] + "";
        if(nums.length == 2)return nums[0] + "/" + nums[1];
        StringBuffer result = new StringBuffer(nums[0] + "/(");
        for(int i = 1;i < nums.length - 1;i++){
        	result.append(nums[i] + "/");
        }
        result.append(nums[nums.length - 1] + ")");
        return result.toString();
    } 
    
    
 * 2017/7/29
 * 565. Array Nesting
 * 
    public int arrayNesting(int[] nums) {
        if(nums.length == 0)return 0;
        int max = Integer.MIN_VALUE;
        for(int i = 0;i < nums.length;i++){
        	nums[i]++;
        }
        for(int i = 0;i < nums.length;i++){
        	if(nums[i] > 0){
        		int temp = arrayMax(nums,i);
        		if(temp > max){
        			max = temp;
        		}
        	}
        }
        return max;
    }
    
    public int arrayMax(int[] nums,int i){
    	int count = 0;
    	int temp = i;
    	while(nums[temp] > 0){
    		nums[temp] = -nums[temp];
    		temp = -nums[temp] - 1;
    		count++;
    	}
    	return count;
    }
 */

/**
 * 2017/7/30
 * 547. Friend Circles
 * 
    public int findCircleNum(int[][] M) {
    	if(M.length == 0 || M[0].length == 0)return 0;
    	int result = 0;
    	Stack<Integer> stack = new Stack<Integer>();
    	for(int i = 0;i < M.length;i++){
    		if(M[i][i] == 1){
    			result++;
    		}
    		else{
    			continue;
    		}
    		for(int j = i + 1;j < M[i].length;j++){
    			if(M[i][j] == 1){
    				M[i][i] = -1;
    				M[j][j] = -1;
    				M[i][j] = -1;
    				stack.add(j);
    			}
    		}
    		while(!stack.isEmpty()){
    			int temp = stack.pop();
    			for(int k = 0;k < temp;k++){
    				if(M[k][temp] == 1){
    					M[k][temp] = -1;
    					M[k][k] = -1;
    					M[temp][temp] = -1;
    					stack.add(k);
    				}
    			}
    			for(int k = temp + 1;k < M[0].length;k++){
    				if(M[temp][k] == 1){
    					M[temp][k] = -1;
    					M[temp][temp] = -1;
    					M[k][k] = -1;
    					stack.add(k);
    				}
    			}
    		}
    	}
    	return result;
    }
    
    
 * 2017/7/30
 * 648. Replace Words
 * 
    public String replaceWords(List<String> dict, String sentence) {
        String[] split = sentence.split(" ");
        for(int i = 0;i < split.length;i++){
        	for(int j = 1;j < split[i].length();j++){
            	if(dict.contains(split[i].substring(0, j))){
            		split[i] = split[i].substring(0, j);
            		break;
            	}
        	}
        }
    	StringBuffer temp = new StringBuffer(split[0]);
    	for(int i = 1;i < split.length;i++){
    		temp.append(" " + split[i]);
    	}
    	return temp.toString();
    }
    
    
 * 2017/7/30
 * 238. Product of Array Except Self
 * 
    public int[] productExceptSelf(int[] nums) {
    	if(nums.length == 1)return nums;
    	int[] result = new int[nums.length];
    	result[0] = 1;
    	for(int i = 1;i < nums.length;i++){
    		result[i] = result[i - 1] * nums[i - 1];
    	}
    	int temp = 1;
    	for(int i = nums.length - 2;i >= 0;i--){
    		temp *= nums[i + 1];
    		result[i] *= temp;
    	}
    	return result;
    }
    
    
 * 2017/7/30
 * 347. Top K Frequent Elements
 * 
    public List<Integer> topKFrequent(int[] nums, int k) {
        Map<Integer,Integer> map = new HashMap<Integer,Integer>();
        for(int i = 0;i < nums.length;i++){
        	map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        List<Integer> result = new ArrayList<Integer>();
        List<Map.Entry<Integer, Integer>> list = new ArrayList<Map.Entry<Integer, Integer>>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<Integer, Integer>>(){
            public int compare(Map.Entry<Integer, Integer> o1, Map.Entry<Integer, Integer> o2) {  
                return o2.getValue() - o1.getValue();  //降序排列
            }  
        });
        for(int i = 0;i < k;i++){
        	result.add(list.get(i).getKey());
        }
        return result;
    }
    
    
 * 2017/7/30
 * 503. Next Greater Element II
 * 
    public int[] nextGreaterElements(int[] nums) {
        int[] result = new int[nums.length];
        int length = nums.length;
        for(int i = 0;i < length;i++){
        	boolean flag = true;
        	for(int j = 1;j < length;j++){
        		if(nums[(i + j) % length] > nums[i]){
        			flag = false;
        			result[i] = nums[(i + j) % length];
        			break;
        		}
        	}
        	if(flag){
        		result[i] = -1;
        	}
        }
        return result;
    }
 */

/**
 * 2017/7/31
 * 382. Linked List Random Node
 * 
public class Solution {
	ListNode head;
     @param head The linked list's head.
    Note that the head is guaranteed to be not null, so it contains at least one node.
    
    public Solution(ListNode head) {
        this.head = head;
    }
    
    Returns a random node's value.
    public int getRandom() {
        int n = 0;
        ListNode temp = head;
        while(temp != null){
        	n++;
        	temp = temp.next;
        }
        temp = head;
        int ran = (int)(Math.random() * n);
        for(int i = 0;i < ran;i++){
        	temp = temp.next;
        }
        return temp.val;
    }
}


 * 2017/7/31
 * 477. Total Hamming Distance
 * 
    public int totalHammingDistance(int[] nums) {
        int result = 0;
        for(int i = 0;i < 32;i++){
        	int count = 0;
        	for(int j = 0;j < nums.length;j++){
        		count += ((nums[j] >> i) & 1);
        	}
        	result += count * (nums.length - count);
        }
        return result;
    }
    
    
 * 2017/7/31
 * 384. Shuffle an Array
 * 
public class Solution {
	public int[] nums;
	public Random random;
	
    public Solution(int[] nums) {
        this.nums = nums;
        random = new Random();
    }
    
    /** Resets the array to its original configuration and return it. 
    public int[] reset() {
        return nums;
    }
    
    /** Returns a random shuffling of the array. 
    public int[] shuffle() {
        int[] result = new int[nums.length];
        for(int i = 0;i < nums.length;i++){
        	int ran = random.nextInt(i + 1);
        	result[i] = result[ran];
        	result[ran] = nums[i];
        }
        return result;
    }
}


 * 2017/7/31
 * 94. Binary Tree Inorder Traversal
 * 
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<Integer>();
        Stack<TreeNode> stack = new Stack<TreeNode>();
        while(root != null || !stack.isEmpty()){
        	while(root != null){
        		stack.add(root);
        		root = root.left;
        	}
        	root = stack.pop();
        	result.add(root.val);
        	root = root.right;
        }
        return result;
    }
    
    
 * 2017/7/31
 * 454. 4Sum II
 * 
    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        Map<Integer,Integer> map = new HashMap<Integer,Integer>();
        int result = 0;
        for(int i = 0;i <A.length;i++){
        	for(int j = 0;j < B.length;j++){
        		int sum = A[i] + B[j];
        		map.put(sum, map.getOrDefault(sum, 0) + 1);
        	}
        }
        for(int i = 0;i < C.length;i++){
        	for(int j = 0;j < D.length;j++){
        		result += map.getOrDefault((-1 * (C[i] + D[j])), 0);
        	}
        }
        return result;
    }
 */

/**
 * 2017/8/1
 * 529. Minesweeper
 * 
public class Solution {
    public char[][] updateBoard(char[][] board, int[] click) {
        if(board[click[0]][click[1]] == 'M'){
        	board[click[0]][click[1]] = 'X';
        	return board;
        }
        DFSUpdateBoard(board,click[0],click[1]);
        return board;
    }
    
    public void DFSUpdateBoard(char[][] board,int row,int col){
    	int rowLen = board.length;
    	int colLen = board[0].length;
    	if(row < 0 || col < 0 || row >= rowLen || col >= colLen || board[row][col] != 'E')return;
    	int num = getNumOfBoard(board,row,col);
    	if(num == 0){
    		board[row][col] = 'B';
    		for(int i = -1;i < 2;i++){
    			for(int j = -1;j < 2;j++){
    				DFSUpdateBoard(board, row + i, col + j);
    			}
    		}
    	}
    	else{
    		board[row][col] = (char)('0' + num);
    	}
    }
    
    public int getNumOfBoard(char[][] board,int row,int col){
    	int count = 0;
    	for(int i = row - 1;i < row + 2;i++){
    		for(int j = col - 1;j < col + 2;j++){
    			if(i < 0 || i >= board.length || j < 0 || j >= board[0].length)continue;
    			if(board[i][j] == 'M' || board[i][j] == 'X')count++;
    		}
    	}
    	return count;
    }
}


 * 2017/8/1
 * 623. Add One Row to Tree
 * 
    public TreeNode addOneRow(TreeNode root, int v, int d) {
        if(d == 1){
        	TreeNode newRoot = new TreeNode(v);
        	newRoot.left = root;
        	newRoot.right = null;
        	return newRoot;
        }
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        int depth = 1;
        queue.add(root);
        while(!queue.isEmpty()){
        	depth++;
        	int size = queue.size();
        	if(depth == d){
        		for(int i = 0;i < size;i++){
        			TreeNode temp = queue.poll();
        			TreeNode Left = new TreeNode(v);
        			Left.left = temp.left;
        			TreeNode Right = new TreeNode(v);
        			Right.right = temp.right;
        			temp.left = Left;
        			temp.right = Right;
        		}
        		return root;
        	}
        	for(int i = 0;i < size;i++){
    			TreeNode temp = queue.poll();
    			if(temp.left != null)queue.add(temp.left);
    			if(temp.right != null)queue.add(temp.right);
        	}
        }
        return root;
    }
    
    
 * 2017/8/1
 * 445. Add Two Numbers II
 * 
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    	Stack<Integer> s1 = new Stack<Integer>();
    	Stack<Integer> s2 = new Stack<Integer>();
    	while(l1 != null){
    		s1.add(l1.val);
    		l1 = l1.next;
    	}
    	while(l2 != null){
    		s2.add(l2.val);
    		l2 = l2.next;
    	}
    	ListNode list = new ListNode(0);
    	int sum = 0;
    	while(!s1.isEmpty() || !s2.isEmpty()){
    		if(!s1.isEmpty()) sum += s1.pop();
    		if(!s2.isEmpty()) sum += s2.pop();
    		list.val = sum % 10;
    		ListNode temp = new ListNode(sum / 10);
    		temp.next = list;
    		list = temp;
    		sum /= 10;
    	}
    	return list.val == 0 ? (list.next) : list;
	}
 */

/**
 * 2017/8/2
 * 646. Maximum Length of Pair Chain
 * 
    public int findLongestChain(int[][] pairs) {
        int result = 0;
        Arrays.sort(pairs,new Comparator<int[]>(){
        	public int compare(int[] a,int[] b){
        		if(a[0] == b[0])return a[1] - b[1];
        		return a[0] - b[0];
        	}
        });
        int[] dp = new int[pairs.length];
        for(int i = 0;i < pairs.length;i++){
        	dp[i] = 1;
        }
        for(int i = 1;i < pairs.length;i++){
        	for(int j = 0;j < i;j++){
        		if(pairs[i][0] > pairs[j][1] && dp[i] < dp[j] + 1){
        			dp[i] = dp[j] + 1;
        		}
        	}
        }
        for(int i = 0;i < dp.length;i++){
        	if(result < dp[i])
        		result = dp[i];
        }
        return result;
    }
    
    
 * 2017/8/2
 * 343. Integer Break
 * 
    public int integerBreak(int n) {
    	int result = 1;
    	if(n == 2)return 1;
        if(n == 3)return 2;
        while(n > 4){
        	n -= 3;
        	result *= 3;
        }
        result *= n;
        return result;
    }
    
    
 * 2017/8/2
 * 498. Diagonal Traverse
 * 
    public int[] findDiagonalOrder(int[][] matrix) {
    	if(matrix.length == 0 || matrix[0].length == 0){
            int[] result = new int[0];
            return result;
        }
    	int row = matrix.length;
    	int col = matrix[0].length;
    	int totalLength = row * col;
        int[] result = new int[totalLength];
        int index = 0;
        int i = 0,j = 0;
        while(index != totalLength){
        	while(i != -1 && j != col){
        		result[index++] = matrix[i][j];
        		i--;
        		j++;
        	}
        	if(index == totalLength)break;
        	if(j != col){
        		i++;
        	}
        	else{
        		i += 2;
        		j--;
        	}
        	while(i != row && j != -1){
        		result[index++] = matrix[i][j];
        		i++;
        		j--;
        	}
        	if(i != row){
        		j++;
        	}
        	else{
        		j += 2;
        		i--;
        	}
        }
        return result;
    }
    
    
 * 2017/8/2
 * 357. Count Numbers with Unique Digits
 * 
    public int countNumbersWithUniqueDigits(int n) {
        if(n == 0)return 1;
        if(n > 10)n = 10;
        if(n == 1)return 10;
        if(n == 2)return 91;
        int temp = 81;
        int result = 91;
        for(int i = 3;i <= n;i++){
        	temp *= (11 - i);
        	result += temp;
        }
        return result;
    }
    
    
 * 2017/8/2
 * 539. Minimum Time Difference
 * 
public class Solution {
    public int findMinDifference(List<String> timePoints) {
    	if(timePoints.size() == 0)return 0;
    	int length = timePoints.size();
    	String[][] str = new String[length][2];
    	for(int i = 0;i < length;i++){
    		str[i] = timePoints.get(i).split(":");
    	}
    	int[][] time = new int[length][2];
    	for(int i = 0;i < length;i++){
    		for(int j = 0;j < 2;j++){
    			time[i][j] = Integer.parseInt(str[i][j]);
    		}
    	}
    	Arrays.sort(time, new Comparator<int[]>() {
 			public int compare(int[] a, int[] b) {
 				//升序
 				if (b[0] == a[0])
 					return a[1] - b[1];
 				return a[0] - b[0];
 			}
 		});
    	int min = Integer.MAX_VALUE;
    	for(int i = 0;i < time.length - 1;i++){
    		int temp = getTime(time[i],time[i + 1]);
    		if(min > temp)min = temp;
    	}
    	int temp = getTime(time[0],time[length - 1]);
		if(min > temp)min = temp;
		return min;
    }
    
    public int getTime(int[] fir,int[] sec){
        if(fir[0] == sec[0])return Math.abs(sec[1] - fir[1]);
        int result = 0;
        if(fir[0] > sec[0]){
        	result = 60 * (fir[0] - sec[0]) + fir[1] - sec[1];
        }
        else{
        	result = 60 * (sec[0] - fir[0]) + sec[1] - fir[1];
        }
        if(result > 720){
        	result = 1440 - result;
        }
        return result;
    }
}


 * 2017/8/2
 * 144. Binary Tree Preorder Traversal
 * 
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<Integer>();
        if(root == null)return result;
        Stack<TreeNode> stack = new Stack<TreeNode>();
        while(root != null || !stack.isEmpty()){
        	while(root != null){
        		stack.push(root);
        		result.add(root.val);
        		root = root.left;
        	}
        	root = stack.pop();
        	root = root.right;
        }
        return result;
    }
 */


package exercise;

public class Solution_July26th_To_August2nd {

}

/**
 * 实现集合排序，继承接口
 */

/**
 * 		Arrays.sort(people, new Comparator<int[]>() {
			public int compare(int[] a, int[] b) {
				if (b[0] == a[0])
					return a[1] - b[1];
				return b[0] - a[0];
			}
		});
		
		List<Map.Entry<Character, Integer>> list = new ArrayList<Map.Entry<Character, Integer>>(map.entrySet());  
        Collections.sort(list, new Comparator<Map.Entry<Character, Integer>>() {  
            //降序排序  
            public int compare(Map.Entry<Character, Integer> o1, Map.Entry<Character, Integer> o2) {  
                return o2.getValue() - o1.getValue();  
            }  
        });  
        
        PriorityQueue<Map.Entry<Character, Integer>> pq = new PriorityQueue<>(
            new Comparator<Map.Entry<Character, Integer>>() {
                public int compare(Map.Entry<Character, Integer> a, Map.Entry<Character, Integer> b) {
                    return b.getValue() - a.getValue();
                }
            }
        );
 */


/** public static void quick(int[] a, int low, int high) {// 递归快排
	if (low < high) {
		quick(a, low, partition(a, low, high) - 1);
		quick(a, partition(a, low, high) + 1, high);
	}
}

public static int partition(int[] a, int low, int high) {
	// 分块方法，在数组a中，对下标从low到high的数列进行划分
	int pivot = a[low];// 把比pivot(初始的pivot=a[low]小的数移动到pivot的左边
	while (low < high) {// 把比pivot大的数移动到pivot的右边
		while (low < high && a[high] >= pivot) {
			high--;
		}
		a[low] = a[high];
		while (low < high && a[low] <= pivot) {
			low++;
		}
		a[high] = a[low];
	}
	a[low] = pivot;
	return low;
	// 返回划分后的pivot的位置
}*/