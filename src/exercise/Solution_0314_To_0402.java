/**
 * @author jhy code from 3.14 to 4.2
 * 27 questions
 */

/**
 * 2018/3/14
 * 165. Compare Version Numbers
 * 
	public int compareVersion(String version1, String version2) {
		String[] levels1 = version1.split("\\.");
		String[] levels2 = version2.split("\\.");

		int length = Math.max(levels1.length, levels2.length);
		for (int i = 0; i < length; i++) {
			Integer v1 = i < levels1.length ? Integer.parseInt(levels1[i]) : 0;
			Integer v2 = i < levels2.length ? Integer.parseInt(levels2[i]) : 0;
			int compare = v1.compareTo(v2);
			if (compare != 0) {
				return compare;
			}
		}

		return 0;
	}
	
 * 2018/3/14
 * 91. Decode Ways
 * 
    public int numDecodings(String s) {
        if(s == null || s.length() == 0) {
            return 0;
        }
        int n = s.length();
        int[] dp = new int[n+1];
        dp[0] = 1;
        dp[1] = s.charAt(0) != '0' ? 1 : 0;
        for(int i = 2; i <= n; i++) {
            int first = Integer.valueOf(s.substring(i-1, i));
            int second = Integer.valueOf(s.substring(i-2, i));
            if(first >= 1 && first <= 9) {
               dp[i] += dp[i-1];  
            }
            if(second >= 10 && second <= 26) {
                dp[i] += dp[i-2];
            }
        }
        return dp[n];
    }
*/

/**
 * 2018/3/15
 * 468. Validate IP Address
 * 
class Solution {
    public String validIPAddress(String IP) {
        if(isValidIPv4(IP))return "IPv4";
        else if(isValidIPv6(IP)) return "IPv6";
        else return "Neither";
    }
    
    public boolean isValidIPv4(String IP){
    	if(IP.length() < 7)return false;
    	if(IP.charAt(0) == '.')return false;
    	if(IP.charAt(IP.length() - 1) == '.')return false;
    	String[] tokens = IP.split("\\.");
    	if(tokens.length != 4)return false;
    	for(String token : tokens){
    		if(!isValidIPv4Token(token))return false;
    	}
    	return true;
    }
    
    public boolean isValidIPv4Token(String token){
    	if(token.startsWith("0") && token.length()>1) return false;
    	try {
			int parse = Integer.parseInt(token);
			if(parse < 0 || parse > 255)return false;
			if(parse == 0 && token.charAt(0) != '0')return false;
		} catch (NumberFormatException e) {
			return false;
		}
    	return true;
    }
    
    public boolean isValidIPv6(String IP){
    	if(IP.length() < 15)return false;
    	if(IP.charAt(0) == ':')return false;
    	if(IP.charAt(IP.length() - 1) == ':')return false;
    	String[] tokens = IP.split(":");
    	if(tokens.length != 8)return false;
    	for(String token : tokens){
    		if(!isValidIPv6Token(token))return false;
    	}
    	return true;
    }
    
    public boolean isValidIPv6Token(String token){
    	if(token.length() > 4 || token.length() == 0)return false;
    	for(char ch : token.toCharArray()){
    		if(!(ch >= '0' && ch <= '9') && !(ch >= 'a' && ch <= 'f') && !(ch >= 'A' && ch <= 'F'))return false;
    	}
    	return true;
    }
}

 * 2018/3/15
 * 151. Reverse Words in a String
 * 
    public String reverseWords(String s) {
        String[] sp = s.split("\\s{1,}");
        StringBuffer sBuffer = new StringBuffer();
        for(int i = sp.length - 1;i >= 0;i--){
        	sBuffer.append(" ").append(sp[i]);
        }
        int left = 0;
        while(left < sBuffer.length() && sBuffer.charAt(left) == ' ')left++;
        int right = sBuffer.length() - 1;
        while(right >= left && sBuffer.charAt(right) == ' ')right--;
        return sBuffer.substring(left, right + 1);
    }
    
 * 2018/3/15
 * 8. String to Integer (atoi)
 * 
    public int myAtoi(String str) {
        int sign = 1;
        int total = 0;
        int index = 0;
        if(str == null || str.length() == 0)return 0;
        while(str.charAt(index) == ' ' && index < str.length())index++;
        if(str.charAt(index) == '+' || str.charAt(index) == '-'){
        	if(str.charAt(index++) == '+')sign = 1;
        	else sign = -1;
        }
        for(;index < str.length();index++){
        	int num = str.charAt(index) - '0';
        	if(num < 0 || num > 9)break;
        	if(total > Integer.MAX_VALUE / 10 || (total == Integer.MAX_VALUE / 10 && num > Integer.MAX_VALUE % 10)){
        		return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
        	}
        	total = total * 10 + num;
        }
        return sign * total;
    }
*/

/**
 * 2018/3/16
 * 127. Word Ladder
 * 
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        int res = 1;
        Queue<String> queue = new LinkedList<>();
        queue.add(beginWord);
        while(!queue.isEmpty()){
        	res++;
        	int size = queue.size();
        	for(int i = 0;i < size;i++){
        		String temp = queue.poll();
        		for(int j = 0;j < wordList.size();j++){
        			String word = wordList.get(j);
        			if(isSimilar(temp, word)){
        				if(endWord.equals(word))return res;
        				queue.add(word);
        				wordList.remove(j);
        				j--;
        			}
        		}
        	}
        }
    	return 0;
    }
    
    public boolean isSimilar(String a, String b){
    	boolean flag = true;
    	for(int i = 0;i < a.length();i++){
    		if(a.charAt(i) != b.charAt(i)){//有一次不等的机会
    			if(flag)flag = false;
    			else return flag;
    		}
    	}
    	return true;
    }
}

 * 2018/3/16
 * 220. Contains Duplicate III
 * 
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        int p = 0; 
        if(nums.length==0) return false; 
        if(k==0) return false; 

        TreeSet<Long> set = new TreeSet<>(); 
        for(int i = 0; i<nums.length; i++){
            Long tmp1 = set.ceiling((long)nums[i]);
            Long tmp2 = set.floor((long)nums[i]); 
            if(tmp1!=null && tmp1 - nums[i]<=t) return true; 
            if(tmp2!=null && nums[i] - tmp2<=t) return true; 
            
            if(set.size()==k) set.remove((long)nums[p++]); 
            set.add((long)nums[i]); 
        }
        return false; 
    }
    
 * 2018/3/16
 * 130. Surrounded Regions
 * 
class Solution {
    public void solve(char[][] board) {
        if(board == null || board.length == 0 || board[0] == null || board[0].length == 0)return;
        m = board.length;
        n = board[0].length;
        for(int i = 0;i < n;i++){
        	if(board[0][i] == 'O')dfs(board, 0, i);
        	if(board[m - 1][i] == 'O')dfs(board, m - 1, i);
        }
        for(int i = 0;i < m;i++){
        	if(board[i][0] == 'O')dfs(board, i, 0);
        	if(board[i][n - 1] == 'O')dfs(board, i, n - 1);
        }
        for(int i = 0;i < m;i++){
        	for(int j = 0;j < n;j++){
        		if(board[i][j] == 'O')board[i][j] = 'X';
        		else if(board[i][j] == '*')board[i][j] = 'O';
        	}
        }
    }
    
    int m;
    int n;
    
    public void dfs(char[][] board, int i, int j){
    	if(i < 0 || i >= m || j < 0 || j >= n)return;
    	if(board[i][j] == 'O')board[i][j] = '*';
    	if(i > 0 && board[i - 1][j] == 'O')dfs(board, i - 1, j);
    	if(j > 0 && board[i][j - 1] == 'O')dfs(board, i, j - 1);
    	if(i < m - 1 && board[i + 1][j] == 'O')dfs(board, i + 1, j);
    	if(j < n - 1 && board[i][j + 1] == 'O')dfs(board, i, j + 1);
    }
}
*/

/**
 * 2018/3/17
 * 166. Fraction to Recurring Decimal
 * 
    public String fractionToDecimal(int numerator, int denominator) {
        if(numerator == 0)return "0";
        StringBuffer res = new StringBuffer();
        //sign
        res.append((numerator > 0) ^(denominator > 0) ? "-" : "");
        long num = Math.abs((long)numerator);
        long div = Math.abs((long)denominator);
        //integral part
        res.append(num / div);
        num %= div;
        if(num == 0)return res.toString();
        Map<Long, Integer> map = new HashMap<>();
        //fraction part
        res.append(".");
        map.put(num, res.length());
        while(num != 0){
        	num *= 10;
        	res.append(num / div);
        	num %= div;
        	if(map.containsKey(num)){
        		res.insert(map.get(num), "(");
        		res.append(")");
        		break;
        	}
        	else{
        		map.put(num, res.length());
        	}
        }
        return res.toString();
    }
    
 * 2018/3/17
 * 464. Can I Win
 * 
class Solution {
    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        int num = (1 + maxChoosableInteger) * maxChoosableInteger / 2;
        if(num < desiredTotal)return false;
        if(desiredTotal <= 0)return true;
        
        Map<Integer, Boolean> map = new HashMap<>();
        boolean[] used = new boolean[maxChoosableInteger + 1];
        return help(desiredTotal, map, used);
    }
    
    public boolean help(int desire, Map<Integer, Boolean> map, boolean[] used){
    	if(desire <= 0)return false;
    	int key = format(used);
    	if(!map.containsKey(key)){
    		for(int i = 1;i < used.length;i++){
    			if(!used[i]){
    				used[i] = true;
    				if(!help(desire - i, map, used)){
    					used[i] = false;
    					map.put(key, true);
    					return true;
    				}
    				used[i] = false;
    			}
    		}
    		map.put(key, false);
    	}
    	return map.get(key);
    }
    
    public int format(boolean[] used){//将used转化为int便于放入map中
    	int res = 0;
    	for(int i = 1;i < used.length;i++){
    		res <<= 1;
    		if(used[i])res |= 1;
    	}
    	return res;
    }
}
*/

/**
 * 2018/3/19
 * 765. Couples Holding Hands
 * 
class Solution {//greedy算法解决问题
    public int minSwapsCouples(int[] row) {
        int res = 0;
        for(int i = 0;i < row.length;i += 2){
        	int num = isCouple(row[i], row[i + 1]);
        	if(num == -1)continue;
        	for(int j = i + 2;j < row.length;j++){
        		if(row[j] == num){
        			swap(row, i + 1, j);
        			res++;
        			break;
        		}
        	}
        }
        return res;
    }
    
    public void swap(int[] row, int i, int j){
    	int temp = row[i];
    	row[i] = row[j];
    	row[j] = temp;
    }
    
    public int isCouple(int fir, int sec){//返回是否是couple，若不是则返回i + 1 应该的值
    	if((fir % 2) == 0 && sec == fir + 1 || (fir % 2) == 1 && fir == sec + 1)return -1;//是couple
    	if(fir % 2 == 0)return fir + 1;
    	else return fir - 1;
    }
}

 * 2018/3/19
 * 773. Sliding Puzzle
 * 
class Solution {//深度优先解决问题
    public int slidingPuzzle(int[][] board) {
        Set<String> set = new HashSet<>();//记录已经过的数组
        String want = "123450";
        if(want.equals(matrixToStr(board)))return 0;
        int res = 1;
        Queue<String> queue = new LinkedList<>();
        queue.add(matrixToStr(board));
        set.add(matrixToStr(board));
        while(!queue.isEmpty()){
        	int size = queue.size();
        	for(int i = 0;i < size;i++){//每一层深度优先搜索
        		int[][] temp =  StringToMat(queue.poll());
        		int row = 0, col = 0;
        		boolean flag = false;//判断是否找到值
				for (row = 0; row < 2; row++) {// 计算为0的位置
					for (col = 0; col < 3; col++) {
						if (temp[row][col] == 0) {
							flag = true;
							break;
						}
					}					
					if(flag)break;
        		}
        		for(int[] d : dir){//不断将0的位置与四周交换
        			int newRow = row + d[0];
        			int newCol = col + d[1];
        			if(isValid(newRow, newCol)){
        				swap(temp, row, col, newRow, newCol);
        				String str = matrixToStr(temp);
        				if(want.equals(str))return res;
        				if(!set.contains(str)){//如果未访问过，则加入到queue中
        					set.add(str);
        					queue.add(str);
        				}
        				swap(temp, row, col, newRow, newCol);
        			}
        		}
        	}
        	res++;
        }
        return -1;
    }
    
    int[][] dir = new int[][]{{0,-1},{0,1},{1,0},{-1,0}};
    
    public boolean isValid(int i, int j){//判断board是否越界
    	return (i == 0 || i == 1) && (j >= 0 && j <= 2);
    }
    
    public String matrixToStr(int[][] board){//board转换为String,便于存储及比较
    	StringBuffer sb = new StringBuffer();
    	for(int i = 0;i < 2;i++){
    		for(int j = 0;j < 3;j++){
    			sb.append(board[i][j]);
    		}
    	}
    	return sb.toString();
    }
    
    public int[][] StringToMat(String format){//String 转化为 board,便于交换元素
    	int[][] board = new int[2][3];
    	int index = 0;
    	for(int i = 0;i < 2;i++){
    		for(int j = 0;j < 3;j++){
    			board[i][j] = (int)(format.charAt(index++) - '0');
    		}
    	}
    	return board;
    }
    
    public void swap(int[][] board, int i, int j, int m, int n){//交换[i][j]与[m][n]
    	int temp = board[i][j];
    	board[i][j] = board[m][n];
    	board[m][n] = temp;
    }
}

 * 2018/3/19
 * 802. Find Eventual Safe States
 * 
public List<Integer> eventualSafeNodes(int[][] graph) {
    	List<Integer> res = new ArrayList<>();
    	if(graph == null || graph.length == 0)return res;
    	Set<Integer> set = new HashSet<>();
    	int size = -1;
    	while(size != set.size()){
    		size = set.size();
    		for(int i = 0;i < graph.length;i++){
    			if(!set.contains(i)){
    				boolean flag = true;
    				for(int j = 0;j < graph[i].length;j++){
    					if(!set.contains(graph[i][j])){
    						flag = false;
    						break;
    					}
    				}
    				if(flag)set.add(i);
    			}
    		}
    	}
    	res.addAll(set);
    	Collections.sort(res);
    	return res;
    }
*/

/**
 * 2018/3/19
 * 801. Minimum Swaps To Make Sequences Increasing
 * 
    public int minSwap(int[] A, int[] B) {
        int len = A.length;
        int[] swap = new int[len];
        int[] not_swap = new int[len];
        swap[0] = 1;
        for(int i = 1;i < len;i++){
        	swap[i] = not_swap[i] = len;
        	if(A[i] > A[i - 1] && B[i] > B[i - 1]){
        		swap[i] = swap[i - 1] + 1;
        		not_swap[i] = not_swap[i - 1];
        	}
        	if(A[i] > B[i - 1] && B[i] > A[i - 1]){
        		swap[i] = Math.min(swap[i], not_swap[i - 1] + 1);
        		not_swap[i] = Math.min(not_swap[i] , swap[i - 1]);
        	}
        }
        return Math.min(swap[len - 1], not_swap[len - 1]);
    }
*/

/**
 * 2018/3/22
 * 324. Wiggle Sort II
 * 
class Solution {
	   public void wiggleSort(int[] nums) {
	        int median = findKthLargest(nums, (nums.length + 1) / 2);
	        int n = nums.length;

	        int left = 0, i = 0, right = n - 1;

	        while (i <= right) {
	            if (nums[newIndex(i,n)] > median) {
	                swap(nums, newIndex(left++,n), newIndex(i++,n));
	            }
	            else if (nums[newIndex(i,n)] < median) {
	                swap(nums, newIndex(right--,n), newIndex(i,n));
	            }
	            else {
	                i++;
	            }
	        }
	    }

    private int newIndex(int index, int n) {
        return (1 + 2*index) % (n | 1);
    }
    
    public int findKthLargest(int[] nums, int k) {
    	if (nums == null || nums.length == 0) return Integer.MAX_VALUE;
        return findKthLargest(nums, 0, nums.length - 1, nums.length - k);
    }    

    public int findKthLargest(int[] nums, int start, int end, int k) {// quick select: kth smallest
    	if (start > end) return Integer.MAX_VALUE;
    	
    	int pivot = nums[end];// Take A[end] as the pivot, 
    	int left = start;
    	for (int i = start; i < end; i++) {
    		if (nums[i] <= pivot) // Put numbers < pivot to pivot's left
    			swap(nums, left++, i);			
    	}
    	swap(nums, left, end);// Finally, swap A[end] with A[left]
    	
    	if (left == k)// Found kth smallest number
    		return nums[left];
    	else if (left < k)// Check right part
    		return findKthLargest(nums, left + 1, end, k);
    	else // Check left part
    		return findKthLargest(nums, start, left - 1, k);
    } 

    void swap(int[] A, int i, int j) {
    	int tmp = A[i];
    	A[i] = A[j];
    	A[j] = tmp;				
    }
}

 * 2018/3/22
 * 145. Binary Tree Postorder Traversal
 * 
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        TreeNode pre = null;
        while(cur != null){
        	stack.add(cur);
        	cur = cur.left;
        }
        while(!stack.isEmpty()){
        	cur = stack.pop();
        	if(cur.right != null && cur.right != pre){
        		stack.add(cur);
        		cur = cur.right;
        		while(cur != null){
        			stack.add(cur);
        			cur = cur.left;
        		}
        	}
        	else{
        		res.add(cur.val);
        		pre = cur;
        	}
        }
        return res;
    }
*/

/**
 * 2018/3/24
 * 793. Preimage Size of Factorial Zeroes Function
 * 
class Solution {
    public int preimageSizeFZF(int K) {
    	return (int)(binarySearch(K) - binarySearch(K - 1));
    }
    
    public long numOfTrailingZeros(long x){
    	long res = 0;
    	while(x > 0){
    		res += x / 5;
    		x = x / 5;
    	}
    	return res;
    }
    
    public long binarySearch(long K){
    	long left = 0;
    	long right = 5 * (K + 1);
    	while(left <= right){
    		long mid = (left + right) / 2;
    		long num = numOfTrailingZeros(mid);
    		if(num <= K){
    			left = mid + 1;
    		}
    		else{
    			right = mid - 1;
    		}
    	}
    	return right;
    }
}
*/

/**
 * 2018/3/25
 * 807. Max Increase to Keep City Skyline
 * 
public int maxIncreaseKeepingSkyline(int[][] grid) {
        if(grid == null || grid.length == 0 || grid[0] == null || grid[0].length == 0)return 0;
        int res = 0;
        int[] left = new int[grid.length];
        for(int i = 0;i < grid.length;i++){
        	int max = 0;
        	for(int j = 0;j < grid[i].length;j++){
        		max = Math.max(max, grid[i][j]);
        	}
        	left[i] = max;
        }
        int[] bottom = new int[grid[0].length];
        for(int i = 0;i < grid[0].length;i++){
        	int max = 0;
        	for(int j = 0;j < grid.length;j++){
        		max = Math.max(max, grid[j][i]);
        	}
        	bottom[i] = max;
        }
        for(int i = 0;i < grid.length;i++){
        	for(int j = 0;j < grid[0].length;j++){
        		res += (Math.min(left[i], bottom[j]) - grid[i][j]);
        	}
        }
        return res;
    }
    
 * 2018/3/25
 * 804. Unique Morse Code Words
 * 
class Solution {
    public int uniqueMorseRepresentations(String[] words) {
        String[] code = new String[]{".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."};
        Map<Character, String> map = new HashMap<>();
        for(char ch = 'a';ch <= 'z';ch++){
        	map.put(ch, code[ch - 'a']);
        }
        Set<String> set = new HashSet<>();
        for(String word : words){
        	set.add(convertToCode(word, map));
        }
        return set.size();
    }
    
    public String convertToCode(String word, Map<Character, String> map){
    	StringBuffer sb = new StringBuffer();
    	for(int i = 0;i < word.length();i++){
    		sb.append(map.get(word.charAt(i)));
    	}
    	return sb.toString();
    }
}

 * 2018/3/25
 * 806. Number of Lines To Write String
 * 
    public int[] numberOfLines(int[] widths, String S) {
    	if(S == null)return new int[]{1,0};
        int line = 1;
        int use = 0;
        for(int i = 0;i < S.length();i++){
        	if(use + widths[S.charAt(i) - 'a'] > 100){
        		line++;
        		use = 0;
        	}
        	use += widths[S.charAt(i) - 'a'];
        }
        return new int[]{line,use};
    }
*/

/**
 * 2018/3/27
 * 805. Split Array With Same Average
 * 
    public boolean splitArraySameAverage(int[] A) {
        int m = A.length / 2;
        int n = A.length;
        int totalNum = 0;
        for(int num : A)totalNum += num;
        boolean flag = false;
        for(int i = 1;i <= m && !flag;i++){
        	if(totalNum * i % n == 0)flag = true;
        }
        if(!flag)return false;
        
        List<Set<Integer>> sum = new ArrayList<>();
        for(int i = 0;i <= m;i++){
        	sum.add(new HashSet<Integer>());
        }
        sum.get(0).add(0);
        for(int i = 0;i < A.length;i++){
			for (int j = m; j >= 1; j--) {
				if (sum.get(j - 1).size() != 0) {
					for (int t : sum.get(j - 1)) {
						sum.get(j).add(t + A[i]);
					}
				}
			}
        }
        for(int i = 1;i <= m;i++){
        	if(totalNum * i % n == 0 && sum.get(i).contains(totalNum * i / n))return true;
        }
        return false;
    }
*/

/**
 * 2018/4/1
 * 52. N-Queens II
 * 
class Solution {
	int res = 0;
	int row = 0;
	int col = 0;
	
    public int totalNQueens(int n) {
    	if(n <= 0)return 0;
    	if(n == 1)return 1;
    	row = n;
    	col = n;
        boolean[][] matrix =  new boolean[n][n];
        help(matrix, 0);
        return res;
    }
    
    public void help(boolean[][] matrix, int start){//matrix--checkerboard, start--from which row to test
    	if(start == row - 1){
    		for(int i = 0;i < col;i++){
    			if(matrix[start][i] == false)res++;
    		}
    		return;
    	}
    	Stack<int[]> stack = new Stack<>();
    	for(int i = 0;i < col;i++){
    		if(matrix[start][i] == false){
    			putIn(matrix, start, i, stack);
    			help(matrix, start + 1);
    			putOut(matrix, stack);
    		}
    	}
    }
    
    public void putIn(boolean[][] matrix, int i, int j, Stack<int[]> stack){//若[i][j]放置queen,保存回溯数据
    	for(int k = 0;k < col;k++){
    		if(matrix[i][k] == false){
    			matrix[i][k] = true;
    			stack.push(new int[]{i,k});
    		}
    	}
    	for(int k = 0;k < row;k++){
    		if(matrix[k][j] == false){
    			matrix[k][j] = true;
    			stack.push(new int[]{k,j});
    		}
    	}
    	for(int k = 0;k < row && isValid(i + k, j + k, row);k++){
    		if(matrix[i + k][j + k] == false){
    			matrix[i + k][j + k] = true;
    			stack.push(new int[]{i + k,j + k});
    		}
    	}
    	for(int k = 0;k < row && isValid(i + k, j - k, row);k++){
    		if(matrix[i + k][j - k] == false){
    			matrix[i + k][j - k] = true;
    			stack.push(new int[]{i + k,j - k});
    		}
    	}
    }
    
    public boolean isValid(int i, int j, int n){
    	return i >= 0 && j >= 0 && i < n && j < n;
    }
    
    public void putOut(boolean[][] matrix, Stack<int[]> stack){
    	while(!stack.isEmpty()){
    		matrix[stack.peek()[0]][stack.peek()[1]] = false;
    		stack.pop();
    	}
    }
}

 * 2018/4/1
 * 811. Subdomain Visit Count
 * 
    public List<String> subdomainVisits(String[] cpdomains) {
        List<String> res = new ArrayList<>();
        Map<String, Integer> map = new HashMap<>();
        for(String domain : cpdomains){
        	int index = domain.indexOf(' ');
        	int num = Integer.valueOf(domain.substring(0,index));
        	String temp = domain.substring(index + 1);
        	map.put(temp, map.getOrDefault(temp, 0) + num);
        	index = domain.indexOf('.', index + 1);
        	while(index != -1){
        		temp = domain.substring(index + 1);
        		map.put(temp, map.getOrDefault(temp, 0) + num);
        		index = domain.indexOf('.', index + 1);
        	}
        }
        for(String s : map.keySet()){
        	res.add(map.get(s) + " " + s);
        }
        return res;
    }
    
 * 2018/4/1
 * 809. Expressive Words
 * 
class Solution {
    public int expressiveWords(String S, String[] words) {
        int res = 0;
        for(String word : words){
        	if(isExpressive(S, word))res++;
        }
        return res;
    }
    
    public boolean isExpressive(String S, String word){
    	int i = 0;
    	int j = 0;
    	while(i < S.length() && j < word.length()){
    		char ichar = S.charAt(i);
    		char jchar = word.charAt(j);
    		if(ichar != jchar)return false;
    		int inum = 0;
    		int jnum = 0;
    		while(i < S.length()){
    			if(S.charAt(i) == ichar){
    				i++;
    				inum++;
    			}
    			else break;
    		}
    		while(j < word.length()){
    			if(word.charAt(j) == jchar){
    				j++;
    				jnum++;
    			}
    			else break;
    		}
    		if(jnum > inum)return false;
    		else if(inum == jnum)continue;
    		else{
    			if(inum >= 3)continue;
    			else return false;
    		}
    	}
    	if(i < S.length() || j < word.length())return false;
    	return true;
    }
}

 * 2018/4/1
 * 808. Soup Servings
 * 
class Solution {
    public double soupServings(int N) {
    	if(N >= 5000)return 1;
    	Map<Integer, Map<Integer, Double>> map = new HashMap<>();
        return help(N, N, map);
    }
    
    public double help(int A, int B, Map<Integer, Map<Integer, Double>> map){
    	if(A <= 0 && B <= 0)return 0.5;
    	if(A <= 0)return 1;
    	if(B <= 0)return 0;
    	if(map.containsKey(A)){
    		if(map.get(A).containsKey(B)){
    			return map.get(A).get(B);
    		}
    	}
    	double res = (help(A - 100, B, map) + help(A - 75, B - 25, map) + help(A - 50, B - 50, map) + help(A - 25, B - 75, map)) / 4;
    	if(map.containsKey(A)){
    		map.get(A).put(B, res);
    	}
    	else{
    		Map<Integer, Double> temp = new HashMap<>();
    		temp.put(B, res);
    		map.put(A, temp);
    	}
    	return res;
    }
}

 * 2018/4/1
 * 810. Chalkboard XOR Game
 * 
    public boolean xorGame(int[] nums) {
        int sum = 0;
        for(int num : nums){
        	sum ^= num;
        }
        return sum == 0 || nums.length % 2 == 0;
    }
*/

/**
 * 2018/3/14
 * 726. Number of Atoms
 * 
class Solution {
	public int[] countNum(String formula, int i){
		int num = 0;
		if(!(i + 1 < formula.length() && formula.charAt(i + 1) >= '0' && formula.charAt(i + 1) <= '9')){//后不接数字
			num = 1;
		}
		else{
			int index = i + 1;
			i++;
			while(i + 1 < formula.length() && formula.charAt(i + 1) >= '0' && formula.charAt(i + 1) <= '9'){
				i++;
			}
			num = Integer.parseInt(formula.substring(index, i+ 1));
		}
		return new int[]{num, i};
	}
	
    public String countOfAtoms(String formula) {
        Map<String, Integer> map = new HashMap<>();
        Stack<Map<String, Integer>> stack = new Stack<>();
        for(int i = 0;i < formula.length();i++){
        	char ch = formula.charAt(i);
        	if(ch >= 'A' && ch <= 'Z'){//若为元素
        		String temp;
        		if(i + 1 < formula.length() && formula.charAt(i + 1) >= 'a' && formula.charAt(i + 1) <= 'z'){//元素有2个字母
        			temp = formula.substring(i, i + 2);
        			i++;
        		}
        		else{
        			temp = formula.substring(i, i + 1);
        		}
        		int tempRes[] = countNum(formula, i);
        		int num = tempRes[0];//记录出现次数
        		i = tempRes[1];
        		map.put(temp, map.getOrDefault(temp, 0) + num);
        	}
        	else if(ch == '('){
        		stack.push(map);
        		map = new HashMap<>();
        	}
        	else if(ch == ')'){
        		int tempRes[] = countNum(formula, i);
        		int num = tempRes[0];//记录出现次数
        		i = tempRes[1];
        		for(String s : map.keySet()){
        			map.put(s, map.get(s) * num);
        		}
        		Map<String, Integer> popMap = stack.pop();
        		for(String s : popMap.keySet()){
        			map.put(s, map.getOrDefault(s, 0) + popMap.get(s));
        		}
        	}
        }
        PriorityQueue<String> queue = new PriorityQueue<>();
        for(String s : map.keySet()){
        	queue.add(s);
        }
        StringBuffer res = new StringBuffer();
        while(!queue.isEmpty()){
        	String temp = queue.poll();
        	res.append(temp);
        	int num = map.get(temp);
        	if(num != 1){
        		res.append(num);
        	}
        }
        return res.toString();
    }
}
*/
package exercise;

import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Stack;







