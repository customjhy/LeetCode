/**
 * @author jhy code from 2.26 to 3.5
 * 28 questions
 */

/**
 * 2018/2/26
 * 721. Accounts Merge
 * 
class Solution {//并查集解决
    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        Map<String, String> parent = new HashMap<>();//保存父结点
        Map<String, String> own = new HashMap<>();//确定结点的主人姓名
        Map<String, TreeSet<String>> contain = new HashMap<>();
        for(List<String> acc : accounts){//并查集初始化，父节点为自己
        	for(int i = 1;i < acc.size();i++){
        		parent.put(acc.get(i), acc.get(i));
        		own.put(acc.get(i), acc.get(0));
        	}
        }
        for(List<String> acc : accounts){//更新并查集，父节点为list的第一个值
        	String p = find(acc.get(1), parent);
        	for(int i = 2;i < acc.size();i++){
        		parent.put(find(acc.get(i), parent), p);//更新结点的父节点
        	}
        }
        for(List<String> acc : accounts){//相同集合统一至一个Set
        	String p = find(acc.get(1), parent);
        	if(!contain.containsKey(p))contain.put(p, new TreeSet<>());
        	for(int i = 1;i < acc.size();i++){
        		contain.get(p).add(acc.get(i));
        	}
        }
        List<List<String>> res = new ArrayList<>();
        for(String str : contain.keySet()){
        	List<String> temp = new ArrayList<>(contain.get(str));
        	temp.add(0, own.get(str));
        	res.add(temp);
        }
        return res;
    }
    
    public String find(String s, Map<String, String> parent) {
		return parent.get(s) == s ? s : find(parent.get(s), parent);
	}
}

 * 2018/2/26
 * 15. 3Sum
 * 
    public List<List<Integer>> threeSum(int[] nums) {
    	List<List<Integer>> res = new ArrayList<>();
    	if(nums.length < 3)return res;
    	Arrays.sort(nums);
    	int sum = 0;
    	int i = 0;
    	while(i < nums.length - 2){
    		int start = i + 1;
    		int end = nums.length - 1;
    		while(start < end){
    			sum = nums[i] + nums[start] + nums[end];
    			if(sum == 0){
    				List<Integer> temp = new ArrayList<>();
    				temp.add(nums[i]);
    				temp.add(nums[start]);
    				temp.add(nums[end]);
    				res.add(temp);
    				start++;
    				while(start < end && nums[start] == nums[start - 1])start++;
    			}
    			else if(sum < 0)start++;
    			else end--;
    		}
    		i++;
    		while(i < nums.length - 2 && nums[i] == nums[i - 1])i++;
    	}
    	return res;
    }
    
 * 2018/2/26
 * 16. 3Sum Closest
 * 
    public int threeSumClosest(int[] nums, int target) {
        int res = nums[0] + nums[1] + nums[nums.length - 1];
        Arrays.sort(nums);
        for(int i = 0;i < nums.length - 2;i++){
        	int start = i + 1;
        	int end = nums.length - 1;
        	while(start < end){
        		int temp = nums[start] + nums[end] + nums[i];
        		if(temp == target)return target;
        		else if(temp < target)start++;
        		else end--;
        		if(Math.abs(res - target) > Math.abs(temp - target))res = temp;
        	}
        }
        return res;
    }
    
 * 2018/2/26
 * 673. Number of Longest Increasing Subsequence
 * 
class Solution {
    public int findNumberOfLIS(int[] nums) {
        if(nums == null || nums.length == 0)return 0;
        if(nums.length == 1)return 1;
    	int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        for(int i = 1;i < nums.length;i++){
        	for(int j = i - 1;j >= 0;j--){
        		if(nums[i] > nums[j]){
        			dp[i] = Math.max(dp[i], dp[j] + 1);
        		}
        	}
        }
        int max = 0;
        for(int i = 0;i < dp.length;i++){
        	if(dp[i] > max){
        		max = dp[i];
        	}
        }
        int res = 0;
        for(int i = 0;i < dp.length;i++){
        	if(dp[i] == max){
        		res += count(dp, nums, i);
        	}
        }
        return res;
    }
    
    public int count(int[] dp, int[] nums, int index){
    	if(dp[index] == 1)return 1;
    	int res = 0;
    	for(int i = index - 1;i >= 0;i--){
    		if(dp[i] + 1 == dp[index] && nums[i] < nums[index]){
    			res += count(dp, nums, i);
    		}
    	}
    	return res;
    }
}

 * 2018/2/26
 * 142. Linked List Cycle II
 * 
    public ListNode detectCycle(ListNode head) {
    	if(head == null)return null;
    	ListNode one = head;
        ListNode two = head;
        boolean flag = false;
        while(one.next != null && two.next != null && two.next.next != null){
        	one = one.next;
        	two = two.next.next;
        	if(one == two){
        		flag = true;
        		break;
        	}
        }
        if(!flag)return null;
        one = head;
        while(one != two){
        	one = one.next;
        	two = two.next;
        }
        return two;
    }
    
 * 2018/2/26
 * 92. Reverse Linked List II
 * 
    public ListNode reverseBetween(ListNode head, int m, int n) {
    	if(head == null)return null;
    	ListNode dommy = new ListNode(0);
    	dommy.next = head;
        ListNode temp = dommy;
        for(int i = 0;i < m - 1;i++){
        	temp = temp.next;
        }
        ListNode tail = temp.next;
        
        ListNode pre = temp.next;
        ListNode cur = pre.next;
        ListNode re;
        for(int i = 0;i < n - m;i++){
        	re = cur.next;
        	cur.next = pre;
        	pre = cur;
        	cur = re;
        }
        temp.next = pre;
        tail.next = cur;
    	
    	return dommy.next;
    }
    
 * 2018/2/26
 * 373. Find K Pairs with Smallest Sums
 * 
    public List<int[]> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<int[]> res = new ArrayList<>();
        if(nums1.length == 0 || nums2.length == 0 || k == 0)return res;
        //最小堆（优先队列）
        //a[0]为num1的值,a[1]为num2的值，a[2]为对应num2的坐标
        PriorityQueue<int[]> queue = new PriorityQueue<>((a,b) -> a[0] + a[1] - b[0] - b[1]);
        for(int i = 0;i < nums1.length && i < k;i++)queue.add(new int[]{nums1[i], nums2[0], 0});
        while(k-- > 0 && !queue.isEmpty()){
        	int[] cur = queue.poll();
        	res.add(new int[]{cur[0], cur[1]});
        	if(cur[2] == nums2.length - 1)continue;
        	queue.add(new int[]{cur[0], nums2[cur[2] + 1], cur[2] + 1});
        }
        return res;
    }
    
 * 2018/2/26
 * 397. Integer Replacement
 * 
	public int integerReplacement(int n) {
		int c = 0;
		while (n != 1) {
			if ((n & 1) == 0) {
				n >>>= 1;
			} else if (n == 3 || ((n >>> 1) & 1) == 0) {
				--n;
			} else {
				++n;
			}
			++c;
		}
		return c;
	}
 */

/**
 * 2018/2/27
 * 236. Lowest Common Ancestor of a Binary Tree
 * 
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    	if(p == null && q == null)return null;
    	else if(p == null)return q;
    	else if(q == null)return p;
        Stack<TreeNode> pStack = new Stack<>();
        Stack<TreeNode> qStack = new Stack<>();
        ancestor(root, p, pStack);
        ancestor(root, q, qStack);
        TreeNode cur = root;
        while(!pStack.isEmpty() && !qStack.isEmpty()){
        	TreeNode ptemp = pStack.pop();
        	TreeNode qtemp = qStack.pop();
        	if(ptemp == qtemp){
        		cur = ptemp;
        	}
        	else{
        		return cur;
        	}
        }
        return cur;
    }
    
    public boolean ancestor(TreeNode root, TreeNode p, Stack<TreeNode> stack){
    	if(root == null)return false;
    	if(root == p){
    		stack.push(root);
    		return true;
    	}
    	else if(ancestor(root.left, p, stack) || ancestor(root.right, p, stack)){
    		stack.push(root);
    		return true;
    	}
    	return false;
    }
}

 * 2018/2/27
 * 576. Out of Boundary Paths
 * 
class Solution {
    public int findPaths(int m, int n, int N, int i, int j) {
        int res = 0;
        int[][] count = new int[m][n];
        count[i][j] = 1;
        int MOD = 1000000007;
        
        int[][] dirs = new int[][]{{0,1}, {0,-1}, {1,0}, {-1,0}};
        for(int k = 0;k < N;k++){
        	int[][] temp = new int[m][n];
        	for(int row = 0;row < m;row++){
        		for(int col = 0;col < n;col++){
        			for(int[] dir : dirs){
        				int r = row + dir[0];
        				int c = col + dir[1];
        				if(r < 0 || r >= m || c < 0 || c >= n){
        					res = (res + count[row][col]) % MOD;
        				}
        				else{
        					temp[r][c] = (temp[r][c] + count[row][col]) % MOD;
        				}
        			}
        		}
        	}
        	count = temp;
        }
        return res;
    }
}

 * 2018/2/27
 * 82. Remove Duplicates from Sorted List II
 * 
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if(head == null)return head;
    	ListNode res = new ListNode(Integer.MAX_VALUE);
        res.next = head;
        ListNode temp = res;
        ListNode cur = temp.next;
        ListNode dup = null;
        while(cur != null){
        	dup = duplicateNode(cur);
        	if(cur == dup){
        		temp.next = cur;
        		cur = cur.next;
        		temp = temp.next;
        	}
        	else{
        		cur = dup;
        	}
        }
        temp.next = cur;
        return res.next;
    }
    
    public ListNode duplicateNode(ListNode temp){//如果有冗余，返回下一个不等的结点，没有则返回原结点
    	if(temp == null || temp.next == null || temp.val != temp.next.val)return temp;
    	while(temp.val == temp.next.val){
    		temp = temp.next;
    		if(temp.next == null)return null;
    	}
    	return temp.next;
    }
}
*/

/**
 * 2018/2/28
 * 775. Global and Local Inversions
 * 
    public boolean isIdealPermutation(int[] A) {
        for(int i = A.length - 1;i > 0;i--){
        	if(i == A[i]){
        		continue;
        	}
        	if(A[i] == i - 1 && A[i - 1] == i){
        		i--;
        		continue;
        	}
        	return false;
        }
        return true;	
    }
    
 * 2018/2/28
 * 55. Jump Game
 * 
    public boolean canJump(int[] nums) {
    	int res = 0;
    	for(int i = 0;i < nums.length;i++){
    		if(i > res)return false;
    		res = Math.max(res, i + nums[i]);
    	}
    	return true;
    }
    
 * 2018/2/28
 * 134. Gas Station
 * 
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int min = gas[0] - cost[0];
        int index = 0;
        int sum = 0;
        for(int i = 0;i < gas.length;i++){
        	sum += gas[i] - cost[i];
        	if(sum < min){
        		min = sum;
        		index = i;
        	}
        }
        if(sum < 0)return -1;
        return (index + 1) % gas.length;
    }
*/

/**
 * 2018/3/1
 * 332. Reconstruct Itinerary
 * 
    public List<String> findItinerary(String[][] tickets) {
        List<String> res = new ArrayList<>();
        if(tickets == null || tickets.length == 0)return res;
        Map<String, PriorityQueue<String>> map = new HashMap<>();//邻接表存储
        for(String[] ticket : tickets){//邻接表初始化
        	String start = ticket[0];
        	if(map.containsKey(start)){
        		map.get(start).add(ticket[1]);
        	}
        	else{
        		PriorityQueue<String> temp = new PriorityQueue<>();
        		temp.add(ticket[1]);
        		map.put(start, temp);
        	}
        }
        Stack<String> stack = new Stack<>();
        stack.add("JFK");
        while(!stack.isEmpty()){
        	while(map.containsKey(stack.peek()) && !map.get(stack.peek()).isEmpty()){
        		stack.add(map.get(stack.peek()).poll());
        	}
        	res.add(0, stack.pop());
        }
        return res;
    }
    
 * 2018/3/1
 * 148. Sort List
 * 
    public ListNode sortList(ListNode head) {
    	//快速排序
    	//找出链表中间点
        if(head == null || head.next == null)return head;
        ListNode slow = head;
        ListNode fast = head;
        while(fast.next != null && fast.next.next != null){
        	slow = slow.next;
        	fast = fast.next.next;
        }
        ListNode leftPart = head;
        ListNode rightPart = slow.next;
        slow.next = null;//左半部分结尾置为null
        ListNode left = sortList(leftPart);
        ListNode right = sortList(rightPart);
        //进行快速排序
        if(left == null)return right;
        if(right == null)return left;
        ListNode res = null;
        ListNode tempRes = null;
        if(left.val < right.val){
        	res = left;
        	left = left.next;
        }
        else{
        	res = right;
        	right = right.next;
        }
        tempRes = res;
        while(left != null && right != null){
        	if(left.val < right.val){
        		res.next = left;
        		res = res.next;
        		left = left.next;
        	}
        	else{
        		res.next = right;
        		res = res.next;
        		right = right.next;
        	}
        }
        if(left == null){
        	res.next = right;
        }
        else{
        	res.next = left;
        }
        return tempRes;
    }

 * 2018/3/1
 * 2. Add Two Numbers
 * 
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if(l1 == null || (l1.val == 0 && l1.next == null))return l2;
        if(l2 == null || (l2.val == 0 && l2.next == null))return l1;
        ListNode cur = null;
        int flag = 0;//表示进位,便于计算使用int型
        //进行初始化
        int i = l1.val + l2.val;
    	l1 = l1.next;
    	l2 = l2.next;
        if(i >= 10){
        	cur = new ListNode(i - 10);
        	flag = 1;
        }
        else{
        	cur = new ListNode(i);
        }
        ListNode res = cur;//结果，保存链表头
        while(l1 != null && l2 != null){
        	int temp = l1.val + l2.val + flag;
        	l1 = l1.next;
        	l2 = l2.next;
        	if(temp >= 10){
        		cur.next = new ListNode(temp - 10);
        		flag = 1;
        	}
        	else{
        		cur.next = new ListNode(temp);
        		flag = 0;
        	}
        	cur = cur.next;
        }
        //l1或l2其中一个不为空
        if(l1 == null){
        	while(l2 != null){
        		int temp = l2.val + flag;
        		l2 = l2.next;
            	if(temp >= 10){
            		cur.next = new ListNode(temp - 10);
            		flag = 1;
            	}
            	else{
            		cur.next = new ListNode(temp);
            		flag = 0;
            	}
            	cur = cur.next;
        	}
        }
        if(l2 == null){
        	while(l1 != null){
        		int temp = l1.val + flag;
        		l1 = l1.next;
            	if(temp >= 10){
            		cur.next = new ListNode(temp - 10);
            		flag = 1;
            	}
            	else{
            		cur.next = new ListNode(temp);
            		flag = 0;
            	}
            	cur = cur.next;
        	}
        }
        //如果最后计算有进位
        if(flag == 1)cur.next = new ListNode(1);
        return res;
    }
*/

/**
 * 2018/3/2
 * 229. Majority Element II
 * 
    public List<Integer> majorityElement(int[] nums) {
        int canditate1 = 0, canditate2 = 1;
        int count1 = 0, count2 = 0;
        for(int num : nums){
        	if(canditate1 == num)count1++;
        	else if(canditate2 == num)count2++;
        	else if(count1 == 0){
        		canditate1 = num;
        		count1 = 1;
        	}
        	else if(count2 == 0){
        		canditate2 = num;
        		count2 = 1;
        	}
        	else{
        		count1--;
        		count2--;
        	}
        }
        count1 = 0;
        count2 = 0;
        for(int num : nums){
        	if(num == canditate1)count1++;
        	if(num == canditate2)count2++;
        }
        List<Integer> res = new ArrayList<>();
        if(count1 > nums.length / 3)res.add(canditate1);
        if(count2 > nums.length / 3)res.add(canditate2);
        return res;
    }
    
 * 2018/3/2
 * 60. Permutation Sequence
 * 
    public String getPermutation(int n, int k) {
        List<Integer> numbers = new ArrayList<>();//存储剩余的目录
        for(int i = 1;i <= n;i++)numbers.add(i);
        int[] factorial = new int[n];//存储阶乘
        factorial[0] = 1;
        for(int i = 1;i < n;i++){
        	factorial[i] = factorial[i - 1] * i;
        }
        StringBuffer temp = new StringBuffer();
        k--;
        for(int i = 1;i <= n;i++){
        	int index = k / factorial[n - i];
        	temp.append(String.valueOf(numbers.get(index)));
        	numbers.remove(index);
        	k = k - index * factorial[n - i];
        }
        return temp.toString();
    }
    
 * 2018/3/2
 * 227. Basic Calculator II
 * 224. Basic Calculator
 * 
class Solution {
    public int calculate(String s) {
    	String suf = infixToSuffix(s);
    	String[] strings = suf.split("\\s+");
    	Stack<Integer> stack = new Stack<>();
    	for(String str : strings){
    		if(str.charAt(0) >= '0' && str.charAt(0) <= '9'){
    			int val = Integer.valueOf(str);
    			stack.add(val);
    		}
    		else{
    			int b = stack.pop();
    			int a = stack.pop();
    			int num = 0;
    			char ch = str.charAt(0);
    			if(ch == '+')num = a + b;
    			else if(ch == '-')num = a - b;
    			else if(ch == '*')num = a * b;
    			else if(ch == '/')num = a / b;
    			stack.add(num);
    		}
    	}
        return stack.peek();
    }
    
    public String infixToSuffix(String s){
    	StringBuffer temp = new StringBuffer();
    	Stack<Character> stack = new Stack<>();
    	for(int i = 0;i < s.length();i++){
    		char ch = s.charAt(i);
    		if(ch == '+' || ch == '-'){
    			while(!stack.isEmpty() && stack.peek() != '('){
    				char pop = stack.pop();
    				temp.append(" ").append(pop);
    			}
    			stack.add(ch);
    		}
    		else if(ch == '*' || ch == '/'){
    			while(!stack.isEmpty() && (stack.peek() == '*' || stack.peek() == '/')){
    				char pop = stack.pop();
    				temp.append(" ").append(pop);
    			}
    			stack.add(ch);
    		}
    		else if(ch == '('){
    			stack.add(ch);
    		}
    		else if(ch == ')'){
    			while(!stack.isEmpty() && stack.peek() != '('){
    				char pop = stack.pop();
    				temp.append(" ").append(pop);
    			}
    			stack.pop();
    		}
    		else if(ch == ' ')continue;
    		else if(ch >= '0' && ch <= '9'){
    			int val = (int)(ch - '0');
    			while(ch >= '0' && ch <= '9' && i != s.length() - 1){
    				ch = s.charAt(i + 1);
    				if(ch >= '0' && ch <= '9'){
    					val = val * 10 + (int)(ch - '0');
    					i++;
    				}
    				else break;
    			}
    			temp.append(" ").append(String.valueOf(val));
    		}
    	}
    	while(!stack.isEmpty()){
			char pop = stack.pop();
			temp.append(" ").append(pop);
    	}
    	return temp.substring(1).toString();
    }
}
*/

/**
 * 2018/3/3
 * 210. Course Schedule II
 * 
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] degree = new int[numCourses];
        List<List<Integer>> list = new ArrayList<>(numCourses);
        for(int i = 0;i < numCourses;i++){
        	list.add(new ArrayList<>());
        }
        //对邻接表初始化
        for(int[] pre : prerequisites){
        	degree[pre[0]]++;
        	list.get(pre[1]).add(pre[0]);
        }
        int[] order = new int[numCourses];//拓扑排序结果
        Queue<Integer> queue = new LinkedList<>();
        for(int i = 0;i < numCourses;i++){
        	if(degree[i] == 0)queue.add(i);
        }
        int visit = 0;//已排序个数
        while(!queue.isEmpty()){
        	int form = queue.poll();
        	order[visit++] = form;
        	for(int node : list.get(form)){
        		degree[node]--;
        		if(degree[node] == 0){
        			queue.add(node);
        		}
        	}
        }
        return visit == numCourses ? order : new int[0];
    }
    
 * 2018/3/3
 * 93. Restore IP Addresses
 * 
class Solution {
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        if(s.length() < 4)return res;
        help(s, new StringBuffer(), 0, 4, res);
        return res;
    }
    
    public void help(String s, StringBuffer gen, int start, int num, List<String> res){//start代表s开始坐标，num为还差几个数
    	if(num == 0){
    		if(start == s.length())res.add(new String(gen.substring(0, gen.length() - 1)));
    		return;
    	}
    	for(int i = start + 1;i < start + 4 && i + num - 2 < s.length();i++){
    		if(Integer.valueOf(s.substring(start,i)) < 256 && !(i != start + 1 && s.charAt(start) == '0')){
    			int index = gen.length();
    			gen.append(s.substring(start,i)).append(".");
    			help(s, gen, i, num - 1, res);
    			gen = gen.delete(index, gen.length());
    		}
    	}
    }
}    

 * 2018/3/3
 * 150. Evaluate Reverse Polish Notation
 * 
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for(String token : tokens){
        	char ch = token.charAt(0);
        	if(ch >= '0' && ch <= '9' || (token.length() > 1 && ch == '-' && token.charAt(1) >= '0' && token.charAt(1) <= '9')){
        		int num = Integer.valueOf(token);
        		stack.add(num);
        	}
        	else{
        		int b = stack.pop();
        		int a = stack.pop();
        		int num = 0;
        		if(ch == '+'){
        			num = a + b;
        		}
        		else if(ch == '-'){
        			num = a - b;
        		}
        		else if(ch == '*'){
        			num = a * b;
        		}
        		else{
        			num = a / b;
        		}
        		stack.add(num);
        	}
        }
        return stack.peek();
    }
*/

/**
 * 2018/3/4
 * 678. Valid Parenthesis String
 * 
class Solution {
    public boolean checkValidString(String s) {//回溯法解决问题
        return help(s, 0, 0);
    }
    
    public boolean help(String s, int start, int count){
    	if(count < 0)return false;
    	for(int i = start;i < s.length();i++){
    		char c = s.charAt(i);
    		if(c == '('){
    			count++;
    		}
    		else if(c == ')'){
    			if(count <= 0)return false;
                count--;
    		}
    		else{
    			return help(s, i + 1, count + 1) || help(s, i + 1, count - 1) || help(s, i + 1, count);
    		}
    	}
    	return count == 0;
    }
}
*/

/**
 * 2018/3/5
 * 456. 132 Pattern
 * 
class Solution {
	class pair{
		int min;
		int max;
		public pair(int min, int max) {
			this.max = max;
			this.min = min;
		}
	}
	
    public boolean find132pattern(int[] nums) {
    	Stack<pair> stack = new Stack<>();
    	for(int num : nums){
    		if(stack.isEmpty() || num < stack.peek().min)stack.add(new pair(num,num));
    		else if(num > stack.peek().min){
    			if(num < stack.peek().max)return true;
    			else{
    				pair last = stack.pop();
    				last.max = num;
    				while(!stack.isEmpty() && num >= stack.peek().max)stack.pop();
    				if(!stack.isEmpty() && num > stack.peek().min)return true;
    				stack.add(last);
    			}
    		}
    	}
    	return false;
    }
}

 * 2018/3/5
 * 31. Next Permutation
 * 
class Solution {
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while(i >= 0 && nums[i] >= nums[i + 1]){
        	i--;
        }
        if(i >= 0){
        	int j = nums.length - 1;
        	while(j >= 0 && nums[j] <= nums[i]){
        		j--;
        	}
        	swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }
    
    public void swap(int[] nums,int i, int j){
    	int temp = nums[i];
    	nums[i] = nums[j];
    	nums[j] = temp;
    }
    
    public void reverse(int[] nums, int start){
        int i = start;
        int j = nums.length - 1;
        while(i < j){
            swap(nums,i,j);
            i++;
            j--;
        }
    }
}

 * 2018/3/5
 * 556. Next Greater Element III
 * 
class Solution {//仿照上面一题，异曲同工
    public int nextGreaterElement(int n) {
        int bit = 0;
        int temp = n;
        while(temp > 0){
        	bit++;
        	temp = temp / 10;
        }
        temp = n;
        int[] nums = new int[bit];
        while(temp > 0){
        	nums[--bit] = temp % 10;
        	temp = temp / 10;
        }
        int i = nums.length - 2;
        while(i >= 0 && nums[i] >= nums[i + 1]){
        	i--;
        }
        if(i == -1)return -1;
        int j = nums.length - 1;
        while(j >= 0 && nums[j] <= nums[i])j--;
        swap(nums, i, j);
        reverse(nums, i + 1);
        int res = 0;
        for(int k = 0;k < nums.length;k++){
        	res = res * 10 + nums[k];
        }
        temp = Integer.MAX_VALUE;
        bit = 0;
        while(temp > 0){
        	bit++;
        	temp = temp / 10;
        }
        if(bit > nums.length)return res;
        int[] compare = new int[bit];
        temp = Integer.MAX_VALUE;
        while(temp > 0){
        	compare[--bit] = temp % 10;
        	temp = temp / 10;
        }
        for(int k = 0;k < nums.length;k++){
        	if(nums[k] > compare[k])return -1;
        }
        return res;
    }
    
    public void swap(int[] nums, int i, int j){
    	int temp = nums[i];
    	nums[i] = nums[j];
    	nums[j] = temp;
    }
    
    public void reverse(int[] nums, int start){
    	int i = start;
    	int j = nums.length - 1;
    	while(i < j){
    		swap(nums, i, j);
    		i++;
    		j--;
    	}
    }
}

 * 2018/3/5
 * 792. Number of Matching Subsequences
 * 
    public int numMatchingSubseq(String S, String[] words) {
        Map<Character, Deque<String>> map = new HashMap<>();
        for(char c = 'a'; c <= 'z';c++){
        	map.put(c, new LinkedList<String>());
        }
        for(String word : words){
        	map.get(word.charAt(0)).addLast(word);
        }
        int res = 0;
        for(char ch : S.toCharArray()){
        	Deque<String> que = map.get(ch);
        	int size = que.size();
        	for(int i = 0;i < size;i++){
        		String temp = que.pollFirst();
        		if(temp.length() == 1)res++;
        		else{
        			map.get(temp.charAt(1)).addLast(temp.substring(1));
        		}
        	}
        }
        return res;
    }
*/
package exercise;

import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

class Solution_0226_To_0305 {

}













