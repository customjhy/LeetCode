/**
 * @author jhy code from 3.6 to 3.13
 * 32 questions
 */

/**
 * 2018/3/6
 * 310. Minimum Height Trees
 * 
class Solution {
//Solution 1 : Brute Solution -- Time Limit Exceeded
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        List<Integer> res = new ArrayList<>();
        //邻接表初始化
        for(int i = 0;i < n;i++){
        	map.put(i, new ArrayList<>());
        }
        for(int[] edge : edges){
        	map.get(edge[0]).add(edge[1]);
        	map.get(edge[1]).add(edge[0]);
        }
        //保留每一个根节点的度
        int[] deg = new int[n];
        for(int i = 0;i < n;i++){
        	deg[i] = degree(map, i);
        }
        //求解结果
        int min = Integer.MAX_VALUE;
        for(int i = 0;i < n;i++){
        	if(min > deg[i])min = deg[i];
        }
        for(int i = 0;i < n;i++){
        	if(deg[i] == min)res.add(i);
        }
        return res;
    }
    
    public int degree(Map<Integer, List<Integer>> map, int root){//计算以root为节点的树高度，广度优先
    	int res = 0;
    	Set<Integer> set = new HashSet<>();
    	Queue<Integer> queue = new LinkedList<>();
    	queue.add(root);
    	while(!queue.isEmpty()){
    		res++;
    		int size = queue.size();
    		for(int i = 0;i < size;i++){
    			int temp = queue.poll();
				if (!set.contains(temp)) {
					set.add(temp);
					for (int num : map.get(temp)) {
						if (!set.contains(num)) {
							queue.add(num);
						}
					}
				}
    		}
    	}
    	return res;
    }
}


class Solution {
	//Solution2 : compare to BFS topological sort -- Accepted
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        List<Integer> leaves = new ArrayList<>();
        if(n == 1){
        	leaves.add(0);
        	return leaves;
        }
        List<Set<Integer>> adj = new ArrayList<>();//邻接矩阵
        for(int i = 0;i < n;i++)adj.add(new HashSet<>());
        for(int[] edge : edges){
        	adj.get(edge[0]).add(edge[1]);
        	adj.get(edge[1]).add(edge[0]);
        }
        for(int i = 0;i < n;i++){
        	if(adj.get(i).size() == 1)leaves.add(i);
        }
        while(n > 2){
        	n -= leaves.size();
        	ArrayList<Integer> newLeaves = new ArrayList<>();
        	for(int leaf : leaves){
        		int temp = adj.get(leaf).iterator().next();
        		adj.get(temp).remove(leaf);
        		if(adj.get(temp).size() == 1)newLeaves.add(temp);
        	}
        	leaves = newLeaves;
        }
        return leaves;
    }
}

 * 2018/3/6
 * 79. Word Search
 * 
class Solution {
    public boolean exist(char[][] board, String word) {
        if(board == null || board.length == 0 || board[0] == null || board[0].length == 0)return false;
        if(word == null || word.length() == 0)return true;
    	boolean res = false;
    	boolean[][] visit = new boolean[board.length][board[0].length];
    	for(int i = 0;i < board.length;i++){
    		for(int j = 0;j < board[0].length;j++){
    			if(board[i][j] == word.charAt(0)){
    				res = res || help(board, i, j, word, 0, visit);
    				if(res)return true;
    			}
    		}
    	}
        return false;
    }
    
    int[][] dibs = new int[][]{{1,0}, {-1,0}, {0,1}, {0,-1}};
    
    public boolean help(char[][] board,int i, int j,//board[i][j]
    					String word, int start, boolean[][] visit){
    	if(start == word.length() || (start == word.length() - 1 && word.charAt(start) == board[i][j]))return true;
    	if(word.charAt(start) != board[i][j])return false;
    	boolean res = false;
    	visit[i][j] = true;
    	for(int[] d : dibs){
    		int row = i + d[0];
    		int col = j + d[1];
    		if(row >= 0 && row < board.length && col >= 0 && col < board[0].length && !visit[row][col]){
    			res = res || help(board, row, col, word, start + 1, visit);
    			if(res)return true;
    		}
    	}
    	visit[i][j] = false;
    	return false;
    }
}
*/

/**
 * 2018/3/7
 * 365. Water and Jug Problem
 * 
class Solution {
    public boolean canMeasureWater(int x, int y, int z) {
        if(x + y < z)return false;
        if(x == z || y == z || x + y == z)return true;
        return (z % GCD(x, y)) == 0;
    }
    
    public int GCD(int a, int b){//greatest common divisor
    	while(b != 0){
    		int temp = b;
    		b = a % b;
    		a = temp;
    	}
    	return a;
    }
}

 * 2018/3/7
 * 43. Multiply Strings
 * 
    public String multiply(String num1, String num2) {
    	int m = num1.length();
    	int n = num2.length();
    	int[] res = new int[m + n];
    	for(int i = m - 1;i >= 0;i--){
    		for(int j = n - 1;j >= 0;j--){
    			int mult = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
    			int p1 = i + j;
    			int p2 = p1 + 1;
    			
    			mult += res[p2];
    			res[p1] = mult % 10;
    			res[p2] = mult / 10;
    		}
    	}
    	StringBuffer temp = new StringBuffer();
    	for(int r : res){
    		if(!(temp.length() == 0 && r == 0))temp.append(r);
    	}
    	return temp.length() == 0 ? "0" : temp.toString();
    }
*/

/**
 * 2018/3/8
 * 794. Valid Tic-Tac-Toe State
 * 
    public boolean validTicTacToe(String[] board) {
        int turns = 0;//0为o走完，1为x走完
        int[] rows = new int[3];//记录每一行x的数量
        int[] cols = new int[3];
        int diag = 0;//对角线--左上到右下
        int antidiag = 0;//斜对角线--右上到左下
        boolean owin = false;
        boolean xwin = false;
        for(int i = 0;i < 3;i++){
        	for(int j = 0;j < 3;j++){
        		if(board[i].charAt(j) == 'X'){
        			turns++;
        			rows[i]++;
        			cols[j]++;
        			if(i == j)diag++;
        			if(i + j == 2)antidiag++;
        		}
        		else if(board[i].charAt(j) == 'O'){
        			turns--;
        			rows[i]--;
        			cols[j]--;
        			if(i == j)diag--;
        			if(i + j == 2)antidiag--;
        		}
        	}
        }
        xwin = rows[0] == 3 || rows[1] == 3 || rows[2] == 3 || 
                cols[0] == 3 || cols[1] == 3 || cols[2] == 3 || 
                diag == 3 || antidiag == 3;
        owin = rows[0] == -3 || rows[1] == -3 || rows[2] == -3 || 
                cols[0] == -3 || cols[1] == -3 || cols[2] == -3 || 
                diag == -3 || antidiag == -3;
        if((owin && turns == 1) || (xwin && turns == 0))return false;
        return ((turns == 0 || turns == 1) && (!(owin && xwin)));
    }
    
 * 2018/3/8
 * 18. 4Sum
 * 
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if(nums.length < 4)return res;
        Arrays.sort(nums);
        int i = 0;
        while(i < nums.length - 3){
        	int j = i + 1;
        	while(j < nums.length - 2){
        		int remain = target - nums[i] - nums[j];
        		int left = j + 1;
        		int right = nums.length - 1;
        		while(left < right){
        			int sum = nums[left] + nums[right];
        			if(sum == remain){
        				List<Integer> list = new ArrayList<>();
        				list.add(nums[i]);
        				list.add(nums[j]);
        				list.add(nums[left]);
        				list.add(nums[right]);
        				res.add(list);
        				left++;
        				while(left < right && nums[left] == nums[left - 1])left++;
        			}
        			else if(sum < remain){
        				left++;
        				while(left < right && nums[left] == nums[left - 1])left++;
        			}
        			else{
        				right--;
        				while(left < right && nums[right] == nums[right + 1])right--;
        			}
        		}
        		j++;
        		while(j < nums.length - 2 && nums[j] == nums[j - 1])j++;
        	}
        	i++;
        	while(i < nums.length - 3 && nums[i] == nums[i - 1])i++;
        }
        return res;
    }
    
 * 2018/3/8
 * 54. Spiral Matrix
 * 
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        if(matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)return res;
    	int up = 0;
        int down = matrix.length - 1;
        int left = 0;
        int right = matrix[0].length - 1;
        while(true){
        	//up行
        	if(left > right)break;
        	for(int i = left;i <= right;i++){
        		res.add(matrix[up][i]);
        	}
        	up++;
        	//right列
        	if(up > down)break;
        	for(int i = up;i <= down;i++){
        		res.add(matrix[i][right]);
        	}
        	right--;
        	//down行
        	if(left > right)break;
        	for(int i = right;i >= left;i--){
        		res.add(matrix[down][i]);
        	}
        	down--;
        	//left列
        	if(up > down)break;
        	for(int i = down;i >= up;i--){
        		res.add(matrix[i][left]);
        	}
        	left++;
        }
        return res;
    }
    
 * 2018/3/8
 * 222. Count Complete Tree Nodes
 * 
class Solution {
	public int height(TreeNode root){
		if(root == null)return -1;
		return 1 + height(root.left);
	}
	
    public int countNodes(TreeNode root) {
    	int h = height(root);
    	if(h < 0)return 0;
    	if(h - 1 == height(root.right)){
    		return (1 << h) + countNodes(root.right);
    	}
    	else{
    		return (1 << (h - 1)) + countNodes(root.left);
    	}
    }
}

 * 2018/3/8
 * 322. Coin Change
 * 
    public int coinChange(int[] coins, int amount) {
    	if(amount == 0)return 0;
    	if(coins == null || coins.length == 0)return -1;
        Set<Integer> set = new HashSet<>();
        for(int coin : coins)set.add(coin);
        if(set.contains(amount))return 1;
        int count = 1;
        while(!set.isEmpty()){
        	count++;
        	Set<Integer> newSet = new HashSet<>();
        	for(int money : set){
        		for(int coin : coins){
        			int temp = money + coin;
        			if(temp > amount)continue;
        			else if(temp == amount)return count;
        			else{
        				newSet.add(temp);
        			}
        		}
        	}
        	set = newSet;
        }
        return -1;
    }
    
 * 2018/3/8
 * 152. Maximum Product Subarray
 * 
    public int maxProduct(int[] nums) {
        int res = nums[0];
        for(int i = 1, max = res, min = res;i < nums.length;i++){
        	//max存储最大值，min存储最小值，因min乘以负数后即成为最大值
        	if(nums[i] < 0){//若[i]为负数，则max与min互换
        		int temp = max;
        		max = min;
        		min = temp;
        	}
        	max = Math.max(nums[i], max * nums[i]);
        	min = Math.min(nums[i], min * nums[i]);
        	res = Math.max(res, max);
        }
        return res;
    }
    
 * 2018/3/8
 * 402. Remove K Digits
 * 
class Solution {
    public String removeKdigits(String num, int k) {
        String temp = num;
        for(int i = 0;i < k;i++){
        	temp = help(temp);
        	if(temp.equals("0"))return temp;
        }
        return temp;
    }
    
    public String help(String num){
    	if(num == null || num.length() < 2)return "0";
    	int i = 0;
    	while(i + 1 < num.length() && num.charAt(i + 1) >= num.charAt(i))i++;
    	if(i == 0){
    		int j = 1;
    		while(j < num.length() && num.charAt(j) == '0')j++;
    		if(j == num.length())return "0";
    		else return num.substring(j);
    	}
    	return num.substring(0,i) + num.substring(i + 1);
    }
}
*/


/**
 * 2018/3/9
 * 6. ZigZag Conversion
 * 
    public String convert(String s, int numRows) {
    	if(numRows <= 1)return s;
        int internal = 2 * numRows - 2;//每一组间隔
        int times = numRows;//未加入到sb中次数
        StringBuffer sb = new StringBuffer();
        int k = 0;
        int len = s.length();
        while(k * internal < len){
        	sb.append(s.charAt(k * internal));
        	k++;
        }
        times--;
        int left = 1;
        int right = internal - 1;
        while(times > 1){
        	k = 0;
        	while(left + k * internal < len){
        		sb.append(s.charAt(left + k * internal));
        		if(right + k * internal < len){
        			sb.append(s.charAt(right + k * internal));
        		}
        		else{
        			break;
        		}
        		k++;
        	}
        	times--;
        	left++;
        	right--;
        }
        k = 0;
        while(left + k * internal < len){
        	sb.append(s.charAt(left + k * internal));
        	k++;
        }
        return sb.toString();
    }
    
 * 2018/3/9
 * 306. Additive Number
 * 
class Solution {
    public boolean isAdditiveNumber(String num) {
        int len = num.length();
        for(int i = 1;i <= len / 2;i++){//i为第一个数字长度
        	for(int j = 1;Math.max(i, j) <= len - i - j;j++){//j为第二个数字长度
        		if(isValid(i, j, num))return true;
        	}
        }
        return false;
    }
    
    public boolean isValid(int i, int j, String num){
    	if(num.charAt(0) == '0' && i > 1)return false;
    	if(num.charAt(i) == '0' && j > 1)return false;
    	String sum;
    	Long x1 = Long.parseLong(num.substring(0, i));
    	Long x2 = Long.parseLong(num.substring(i, i + j));
    	for(int start = i + j; start < num.length();start += sum.length()){
    		x2 = x1 + x2;//变为前两个数之和
    		x1 = x2 - x1;//变为第二个数
    		sum = x2.toString();//前两个数之和
    		if(!num.startsWith(sum, start))return false;//num从start以后的字符串是否以sum为前缀
    	}
    	return true;
    }
}
*/

/**
 * 2018/3/10
 * 722. Remove Comments
 * 
    public List<String> removeComments(String[] source) {
        List<String> res = new ArrayList<>();
        StringBuffer sb = new StringBuffer();
        boolean mode = false;//是否为"/ * * /"模式
        for(String sou : source){
        	for(int i = 0;i < sou.length();i++){
        		if(mode){
        			if(sou.charAt(i) == '*' && i < sou.length() - 1 && sou.charAt(i + 1) == '/'){
        				mode = false;
        				i++;
        			}
        		}
        		else{
        			if(sou.charAt(i) == '/' && i < sou.length() - 1 && sou.charAt(i + 1) == '/'){
        				break;
        			}
        			else if(sou.charAt(i) == '/' && i < sou.length() - 1 && sou.charAt(i + 1) == '*'){
        				mode = true;
        				i++;
        			}
        			else{
        				sb.append(sou.charAt(i));
        			}
        		}
        	}
        	if(!mode && sb.length() != 0){
        		res.add(sb.toString());
        		sb = new StringBuffer();
        	}
        }
        return res;
    }
    
 * 2018/3/10
 * 143. Reorder List
 * 
class Solution {
    public void reorderList(ListNode head) {
        if(head == null || head.next == null || head.next.next == null)return;
        ListNode fast = head;
        ListNode slow = head;
        while(fast.next != null && fast.next.next != null){
        	slow = slow.next;
        	fast = fast.next.next;
        }
        ListNode temp = slow.next;
        slow.next = null;
        ListNode list2 = reverseList(temp);
        ListNode list1 = head;
        ListNode res = new ListNode(0);
        while(list2 != null){
        	res.next = list1;
        	res = res.next;
        	list1 = list1.next;
        	res.next = list2;
        	res = res.next;
        	list2 = list2.next;
        }
        if(list1 != null)res.next = list1;
    }
	
	public ListNode reverseList(ListNode head){//对链表进行翻转
		if(head == null)return head;
		if(head.next == null)return head;
		ListNode flag = head;
		ListNode p = head.next;
		ListNode temp;
		head.next = null;
		while(p.next != null){
			temp = p;
			p = temp.next;
			temp.next = flag;
			flag = temp;
		}
		p.next = flag;
		return p;
	}
}
*/

/**
 * 2018/3/11
 * 71. Simplify Path
 * 
    public String simplifyPath(String path) {
        Deque<String> stack = new LinkedList<>();
        Set<String> skip = new HashSet<>(Arrays.asList("..", ".", ""));
        for(String dir : path.split("/")){
        	if(dir.equals("..") && !stack.isEmpty())stack.pop();
        	else if(!skip.contains(dir))stack.push(dir);
        }
        String res = "";
        for(String s : stack){
        	res = "/" + s + res;
        }
        return res.length() == 0 ? "/" : res;
    }
    
 * 2018/3/11
 * 777. Swap Adjacent in LR String
 * 
class Solution {
    public boolean canTransform(String start, String end) {
        char[] s = start.toCharArray();
        char[] e = end.toCharArray();
        if (s.length != e.length) return false;
        for (int i = 0; i < s.length; i++) {
            if (s[i] == e[i]) continue;
            if (s[i] != 'X' && e[i] != 'X') return false;
            if (s[i] == 'L' && e[i] == 'X') return false;
            if (s[i] == 'X' && e[i] == 'R') return false;
            if (s[i] == 'R' && e[i] == 'X') {
                int nextR = findNext(s, i+1, 'X', 'R');
                if (nextR == -1) return false;
                else swap(s, i, nextR);
            } else if (s[i] == 'X' && e[i] == 'L') {
                int nextL = findNext(s, i+1, 'L', 'X');
                if (nextL == -1) return false;
                else swap(s, i, nextL);
            }
            
        }
        return true;
    }
    private int findNext(char[] s, int startIdx, char target, char skip) {
        for (int i = startIdx; i < s.length; i++) {
            if (s[i] == target) return i;
            else if (s[i] == skip) continue;
            else return -1;
        }
        return -1;
    }
    private void swap(char[] s, int i, int j) {
        char tmp = s[i];
        s[i] = s[j];
        s[j] = tmp;
    }
}

 * 2018/3/11
 * 796. Rotate String
 * 
    public boolean rotateString(String A, String B) {
        if(A == null || B == null || A.length() != B.length())return false;
        if(A.equals(B))return true;
        for(int i = 0;i < A.length();i++){
        	String temp = A.substring(1) + A.charAt(0);
        	if(temp.equals(B))return true;
        	A = temp;
        }
        return false;
    }
    
 * 2018/3/11
 * 797. All Paths From Source to Target
 * 
class Solution {
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
    	List<List<Integer>> res = new ArrayList<>();
    	int len = graph.length;
    	boolean[] visit = new boolean[len];
    	help(res, new ArrayList<>(), graph, visit, 0, len - 1);
    	return res;
    }
    
    public void help(List<List<Integer>> res, List<Integer> resTemp, int[][] graph, boolean[] visit, int start, int target){
    	if(start == target){
    		resTemp.add(start);
    		res.add(new ArrayList<>(resTemp));
    		resTemp.remove(resTemp.size() - 1);
    		return;
    	}
    	for(int node : graph[start]){
    		if(!visit[node]){
    			visit[node] = true;
    			resTemp.add(start);
    			help(res, resTemp, graph, visit, node, target);
    			resTemp.remove(resTemp.size() - 1);
    			visit[node] = false;
    		}
    	}
    }
}

 * 2018/3/11
 * 795. Number of Subarrays with Bounded Maximum
 * 
    public int numSubarrayBoundedMax(int[] A, int L, int R) {
    	int res = 0;
    	int j = 0,count = 0;
    	for(int i = 0;i < A.length;i++){
    		if(A[i] >= L && A[i] <= R){
    			count = i - j + 1;
    			res += count;
    		}
    		else if(A[i] < L){
    			res += count;
    		}
    		else{
    			j = i + 1;
    			count = 0;
    		}
    	}
    	return res;
    }
*/

/**
 * 2018/3/12
 * 798. Smallest Rotation with Highest Score
 * 
    public int bestRotation(int[] A) {
        int N = A.length;
    	int[] change = new int[N];
        for(int i = 0;i < A.length;i++){
        	change[(i - A[i] + 1 + N) % N] -= 1;
        }
        int max = 0;
        for(int i = 1;i < N;i++){
        	change[i] += change[i - 1] + 1;
        	if(change[i] > change[max]){
        		max = i;
        	}
        }
        return max;
    }
    
 * 2018/3/12
 * 5. Longest Palindromic Substring
 * 
    public String longestPalindrome(String s) {
    	if(s == null || s.length() < 2)return s;
    	int N = s.length();
    	String res = null;
    	boolean[][] dp = new boolean[N][N];
    	for(int i = N - 1;i >= 0;i--){
    		for(int j = i;j < N;j++){
    			dp[i][j] = ((s.charAt(i) == s.charAt(j)) && (j - i < 3 || dp[i + 1][j - 1]));
    			if(dp[i][j] && (res == null || (j - i + 1) > res.length()) ){
    				res = s.substring(i, j + 1);
    			}
    		}
    	}
    	return res;
    }
    
 * 2018/3/12
 * 754. Reach a Number
 * 
    public int reachNumber(int target) {
    	target = Math.abs(target);
        int step = 0;
        int sum = 0;
        while(sum < target){
        	step++;
        	sum += step;
        }
        while((sum - target) % 2 != 0){
        	step++;
        	sum += step;
        }
        return step;
    }
    
 * 2018/3/12
 * 799. Champagne Tower
 * 
    public double champagneTower(int poured, int query_row, int query_glass) {
        double[] dp = new double[101];
        dp[0] = poured;
        for(int row = 1;row <= query_row;row++){
        	for(int i = row;i >= 0;i--){
        		dp[i] = Math.max(0, (dp[i] - 1) / 2);
        		dp[i + 1] += dp[i];
        	}
        }
        return Math.min(1, dp[query_glass]);
    }
*/

/**
 * 2018/3/13
 * 138. Copy List with Random Pointer
 * 
public class Solution {
	class RandomListNode {
		int label;
		RandomListNode next, random;

		RandomListNode(int x) {
			this.label = x;
		}
	};

	public RandomListNode copyRandomList(RandomListNode head) {
		Map<RandomListNode, RandomListNode> map = new HashMap<>();
		RandomListNode temp = head;
		while(temp != null){
			map.put(temp, new RandomListNode(temp.label));
			temp = temp.next;
		}
		temp = head;
		while(temp != null){
			map.get(temp).next = map.get(temp.next);
			map.get(temp).random = map.get(temp.random);
			temp = temp.next;
		}
		return map.get(head);
	}
}

 * 2018/3/13
 * 133. Clone Graph
 * 
public class Solution {
	class UndirectedGraphNode {
		int label;
		List<UndirectedGraphNode> neighbors;
		UndirectedGraphNode(int x) {
			label = x;
			neighbors = new ArrayList<UndirectedGraphNode>();
		}
	};

	public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
		if(node == null)return node;
		UndirectedGraphNode newNode = new UndirectedGraphNode(node.label);
		Map<Integer, UndirectedGraphNode> map = new HashMap<>();
		map.put(newNode.label, newNode);
		
		LinkedList<UndirectedGraphNode> queue = new LinkedList<>();
		queue.add(node);
		while(!queue.isEmpty()){
			UndirectedGraphNode temp = queue.pop();
			for(UndirectedGraphNode neighbour : temp.neighbors){
				if(!map.containsKey(neighbour.label)){
					map.put(neighbour.label, new UndirectedGraphNode(neighbour.label));
					queue.add(neighbour);
				}
				map.get(temp.label).neighbors.add(map.get(neighbour.label));
			}
		}
		return newNode;
	}
}

 * 2018/3/13
 * 179. Largest Number
 * 
    public String largestNumber(int[] nums) {
    	if(nums == null || nums.length == 0)return null;
    	String[] num = new String[nums.length];
    	for(int i = 0;i < num.length;i++){
    		num[i] = String.valueOf(nums[i]);
    	}
        Arrays.sort(num,new Comparator<String>() {
			public int compare(String o1, String o2) {
				// TODO 自动生成的方法存根
				String s1 = o1 + o2;
				String s2 = o2 + o1;
				return s1.compareTo(s2);
			}
		});
        if(num[num.length - 1].charAt(0) == '0')return "0";
        StringBuffer res = new StringBuffer();
        for(int i = num.length - 1;i >= 0;i--){
        	res.append(num[i]);
        }
        return res.toString();
    }
    
 * 2018/3/13
 * 61. Rotate List
 * 
    public ListNode rotateRight(ListNode head, int k) {
    	if(head == null || head.next == null)return head;
    	ListNode cur = head;
    	int size = 1;
    	while(cur.next != null){
    		cur = cur.next;
    		size++;
    	}
    	cur.next = head;
    	k = k % size;
    	k = size - k;
    	while(k > 0){
    		head = head.next;
    		cur = cur.next;
    		k--;
    	}
    	cur.next = null;
    	return head;
    }
    
 * 2018/3/13
 * 98. Validate Binary Search Tree
 * 
	public boolean isValidBST(TreeNode root) {
		if (root == null)
			return true;
		Stack<TreeNode> stack = new Stack<>();
		TreeNode pre = null;
		while (root != null || !stack.isEmpty()) {
			while (root != null) {
				stack.push(root);
				root = root.left;
			}
			root = stack.pop();
			if (pre != null && root.val <= pre.val)
				return false;
			pre = root;
			root = root.right;
		}
		return true;
	}
	
 * 2018/3/13
 * 3. Longest Substring Without Repeating Characters
 * 
    public int lengthOfLongestSubstring(String s) {
    	if(s == null || s.length() == 0)return 0;
        HashSet<Character> set = new HashSet<>();
        int left = 0, right = 0, len = 1;
        set.add(s.charAt(0));
        for(right = 1;right < s.length();right++){
        	while(set.contains(s.charAt(right))){
        		set.remove(s.charAt(left++));
        	}
        	set.add(s.charAt(right));
        	if(right - left + 1 > len)len = right - left + 1;
        }
        return len;
    }
    
 * 2018/3/13
 * 523. Continuous Subarray Sum
 * 
    public boolean checkSubarraySum(int[] nums, int k) {
    	if(nums == null || nums.length < 2)return false;
    	for(int i = 0;i < nums.length - 1;i++){
    		if(nums[i] == 0 && nums[i + 1] == 0)return true;
    	}
    	if(k == 0)return false;
    	k = Math.abs(k);
    	Map<Integer, Integer> map = new HashMap<>();
    	int sum = 0;
    	map.put(0, -1);
    	for(int i = 0;i < nums.length;i++){
    		sum += nums[i];
    		for(int j = (sum / k) * k;j > 0;j -= k){
    			if(map.containsKey(sum - j) && i - map.get(sum - j) > 1)return true;
    		}
    		if(!map.containsKey(sum))map.put(sum, i);
    	}    			
        return false;
    }
    
 * 2018/3/13
 * 29. Divide Two Integers
 * 
class Solution {
    public int divide(int dividend, int divisor) {
        int sign = 1;
        if((dividend < 0 && divisor > 0) || (dividend > 0 && divisor < 0))sign = -1;
        long ldividend = Math.abs((long)dividend);
        long ldivisor = Math.abs((long)divisor);
        if(ldivisor == 0)return Integer.MAX_VALUE;
        if(ldividend == 0 || ldividend < ldivisor)return 0;
        
        long res = help(ldividend, ldivisor);
        int ans = 0;
        if(res > Integer.MAX_VALUE){
        	if(sign == 1)ans = Integer.MAX_VALUE;
        	else ans = Integer.MIN_VALUE;
        	return ans;
        }
        ans = (int)(sign * res);
        return ans;
    }
    
    public long help(long ldividend, long ldivisor){
    	if(ldividend < ldivisor)return 0;
    	long sum = ldivisor;
    	long res = 1;
    	while(sum + sum <= ldividend){
    		res += res;
    		sum += sum;
    	}
    	return res + help(ldividend - sum, ldivisor);
    }
}
*/

package exercise;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Stack;









