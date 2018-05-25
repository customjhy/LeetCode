/**
 * @author jhy code from 4.15 to 4.24
 * 27 questions
 */

/**
 * 2018/4/15
 * 736. Parse Lisp Expression
 * 
class Solution {
    ArrayList<Map<String, Integer>> scope;
    public Solution() {
        scope = new ArrayList();
        scope.add(new HashMap());
    }

    public int evaluate(String expression) {
        scope.add(new HashMap());
        int ans = evaluate_inner(expression);
        scope.remove(scope.size() - 1);
        return ans;
    }

    public int evaluate_inner(String expression) {
        if (expression.charAt(0) != '(') {
            if (Character.isDigit(expression.charAt(0)) || expression.charAt(0) == '-')
                return Integer.parseInt(expression);
            for (int i = scope.size() - 1; i >= 0; --i) {
                if (scope.get(i).containsKey(expression))
                    return scope.get(i).get(expression);
            }
        }

        List<String> tokens = parse(expression.substring(
                expression.charAt(1) == 'm' ? 6 : 5, expression.length() - 1));
        if (expression.startsWith("add", 1)) {
            return evaluate(tokens.get(0)) + evaluate(tokens.get(1));
        } else if (expression.startsWith("mult", 1)) {
            return evaluate(tokens.get(0)) * evaluate(tokens.get(1));
        } else {
            for (int j = 1; j < tokens.size(); j += 2) {
                scope.get(scope.size() - 1).put(tokens.get(j-1), evaluate(tokens.get(j)));
            }
            return evaluate(tokens.get(tokens.size() - 1));
        }
    }

    public List<String> parse(String expression) {
        List<String> ans = new ArrayList();
        int bal = 0;
        StringBuilder buf = new StringBuilder();
        for (String token: expression.split(" ")) {
            for (char c: token.toCharArray()) {
                if (c == '(') bal++;
                if (c == ')') bal--;
            }
            if (buf.length() > 0) buf.append(" ");
            buf.append(token);
            if (bal == 0) {
                ans.add(new String(buf));
                buf = new StringBuilder();
            }
        }
        if (buf.length() > 0)
            ans.add(new String(buf));

        return ans;
    }
}

 * 2018/4/15
 * 819. Most Common Word
 * 
    public String mostCommonWord(String paragraph, String[] banned) {
        Map<String, Integer> count = new HashMap<>();
        int i = 0;
        while(i < paragraph.length() && !(paragraph.charAt(i) >= 'a' && paragraph.charAt(i) <= 'z' || paragraph.charAt(i) >= 'A' && paragraph.charAt(i) <= 'Z' )){
        	i++;
        }
        int j = i;
        while(j < paragraph.length()){
            while(j < paragraph.length() && (paragraph.charAt(j) >= 'a' && paragraph.charAt(j) <= 'z' || paragraph.charAt(j) >= 'A' && paragraph.charAt(j) <= 'Z' )){
            	j++;
            }
            String word = paragraph.substring(i, j).toLowerCase();
        	count.put(word, count.getOrDefault(word, 0) + 1);
            i = j + 1;
            while(i < paragraph.length() && !(paragraph.charAt(i) >= 'a' && paragraph.charAt(i) <= 'z' || paragraph.charAt(i) >= 'A' && paragraph.charAt(i) <= 'Z' )){
            	i++;
            }
            j = i;
        }
        Set<String> ban = new HashSet<>();
        for(String string : banned)ban.add(string.toLowerCase());
        int num = 0;
        String res = null;
        for(String str : count.keySet()){
        	if(count.get(str) > num && !ban.contains(str)){
        		num = count.get(str);
        		res = str;
        	}
        }
        return res;
    }
*/

/**
 * 2018/4/16
 * 817. Linked List Components
 * 
    public int numComponents(ListNode head, int[] G) {
        Set<Integer> set = new HashSet<>();
        for(int num : G)set.add(num);
        boolean flag = false;
        int res = 0;
        while(head != null){
        	if(!flag && set.contains(head.val)){
        		res++;
        		flag = true;
        	}
        	else if(flag && !set.contains(head.val)){
        		flag = false;
        	}
        	head = head.next;
        }
        return res;
    }
    
 * 2018/4/16
 * 816. Ambiguous Coordinates
 * 
class Solution {
    public List<String> ambiguousCoordinates(String S) {
        List<String> res = new ArrayList<>();
        String numString = S.substring(1,S.length() - 1);
        for(int i = 1;i < numString.length();i++){
        	for(String str1 : permutation(numString.substring(0, i))){
        		for(String str2 : permutation(numString.substring(i))){
        			res.add("(" + str1 + ", " + str2 + ")");
        		}
        	}
        }
    	return res;
    }
    
    public List<String> permutation(String S){
    	List<String> res = new ArrayList<>();
    	if(S.length() == 1){
    		res.add(S);
    		return res;
    	}
    	int length = S.length();
    	if(S.charAt(0) == '0'){
    		if(!(S.charAt(length - 1) == '0')){
    			res.add(S.charAt(0) + "." + S.substring(1));
    		}
    	}
    	else{
    		if(S.charAt(length - 1) == '0'){
    			res.add(S);
    		}
    		else{
    			for(int i = 1;i < length;i++){
    				res.add(S.substring(0, i) + "." + S.substring(i));
    			}
    			res.add(S);
    		}
    	}
    	return res;
    }
}
*/

/**
 * 2018/4/17
 * 301. Remove Invalid Parentheses
 * 
class Solution {
    public List<String> removeInvalidParentheses(String s) {
        List<String> res = new ArrayList<>();
        remove(s, res, 0, 0, new char[]{'(', ')'});
        return res;
    }
    
    public void remove(String s, List<String> res, int lastI, int lastJ, char[] remark){
    	for(int stack = 0, i = lastI;i < s.length();i++){
    		if(s.charAt(i) == remark[0])stack++;
    		if(s.charAt(i) == remark[1])stack--;
    		if(stack >= 0)continue;
    		for(int j = lastJ;j <= i;j++){
    			if(s.charAt(j) == remark[1] && (j == lastJ || s.charAt(j - 1) != remark[1])){
    				remove(s.substring(0, j) + s.substring(j + 1), res, i, j, remark);
    			}
    		}
    		return;
    	}
    	String reversed = new StringBuffer(s).reverse().toString();
    	if(remark[0] == '('){
    		remove(reversed, res, 0, 0, new char[]{')', '('});
    	}
    	else{
    		res.add(reversed);
    	}
    }
}

 * 2018/4/17
 * 699. Falling Squares
 * 
    public List<Integer> fallingSquares(int[][] positions) {
        int[] height = new int[positions.length];
        for(int i = 0;i < positions.length;i++){
        	int left = positions[i][0];
        	int size = positions[i][1];
        	int right = left + size;
        	height[i] += size;
        	for(int j = i + 1;j < positions.length;j++){
        		int newLeft = positions[j][0];
        		int newRight = newLeft + positions[j][1];
        		if(newLeft < right && newRight > left){
        			height[j] = Math.max(height[j], height[i]);
        		}
        	}
        }
        List<Integer> res = new ArrayList<>();
        int max = -1;
        for(int hei : height){
        	max = Math.max(max, hei);
        	res.add(max);
        }
        return res;
    }
*/

/**
 * 2018/4/18
 * 546. Remove Boxes
 * 
class Solution {
    public int removeBoxes(int[] boxes) {
        int len = boxes.length;
        int[][][] count = new int[len][len][len];
        return help(count, boxes, 0, len - 1, 0);
    }
    
    public int help(int[][][] count,int[] boxes, int l, int r, int k){
    	if(l > r)return 0;
    	if(count[l][r][k] != 0)return count[l][r][k];
    	while(l < r && boxes[r - 1] == boxes[r]){
    		k++;
    		r--;
    	}
    	int max = help(count, boxes, l, r - 1, 0) + (k + 1) * (k + 1);
    	for(int i = l;i < r;i++){
    		if(boxes[i] == boxes[r]){
    			max = Math.max(max, help(count, boxes, l, i, k + 1) + help(count, boxes, i + 1, r - 1, 0));
    		}
    	}
    	count[l][r][k] = max;
    	return max;
    }
}
*/

/**
 * 2018/4/19
 * 315. Count of Smaller Numbers After Self
 * 
class Solution {
	class Node{
		public int dup = 1;//记录重复的次数
		public int sum, val;
		public Node left;
		public Node right;
		public Node(int v, int s){
			val = v;
			sum = s;
		}
	}
	
    public List<Integer> countSmaller(int[] nums) {
        int[] res = new int[nums.length];
        Node root = null;
        for(int i = nums.length - 1;i >= 0;i--){
        	root = buildTree(root, nums[i], res, i, 0);
        }
        List<Integer> resList = new ArrayList<>();
        for(int n : res)resList.add(n);
        return resList;
    }
    
    public Node buildTree(Node root, int num, int[] res, int i, int preSum){
    	if(root == null){
    		root = new Node(num, 0);
    		res[i] = preSum;
    		return root;
    	}
    	if(root.val == num){
    		root.dup++;
    		res[i] = preSum + root.sum;
    	}else if(root.val > num){
    		root.sum++;
    		root.left = buildTree(root.left, num, res, i, preSum);
    	}else{
    		root.right = buildTree(root.right, num, res, i, preSum + root.sum + root.dup);
    	}
    	return root;
    }
}

//Binary indexed tree 此方法对于有重复的元素如ans = [-1,-1]不适用，因map会覆盖排序的值
class Solution {
    public List<Integer> countSmaller(int[] nums) {
    	List<Integer> res = new ArrayList<>();
        if(nums == null || nums.length == 0)return res;
    	int len = nums.length;
    	BinaryIndexedTree bit = new BinaryIndexedTree();
    	bit.N = len;
    	bit.BIT = new int[len + 1];
    	int[] sortedNum = new int[len];
    	for(int i = 0;i < len;i++)sortedNum[i] = nums[i];
    	Arrays.sort(sortedNum);
    	Map<Integer, Integer> map = new HashMap<>();//将nums[i]与其排序后位置做映射
    	for(int i = 0;i < len;i++)map.put(sortedNum[i], i + 1);
    	for(int i = len - 1;i >= 0;i--){
    		res.add(0, bit.sum(map.get(nums[i])));
    		bit.update(map.get(nums[i]), 1);
    	}
    	return res;
    }
}

class BinaryIndexedTree{
	public int N;
	public int[] BIT;
	public void update(int k, int val){//[k]从pre改为val
		while(k <= N){
			BIT[k] += val;
			k += (k & -k);
		}
	}
	public int sum(int k){
		int sum = 0;
		while(k > 0){
			sum += BIT[k];
			k -= (k & -k);
		}
		return sum;
	}
}

 * 2018/4/19
 * 782. Transform to Chessboard
 * 
    public int movesToChessboard(int[][] board) {
        int N = board.length;
        int rowSum = 0, colSum = 0, rowSwap = 0, colSwap = 0;
        for(int i = 0;i < N;i++){
        	for(int j = 0;j < N;j++){
        		if((board[0][0] ^ board[i][0] ^ board[0][j] ^ board[i][j]) == 1)return -1;
        	}
        }
        for(int i = 0;i < N;i++){
        	rowSum += board[i][0];
        	colSum += board[0][i];
        	if(board[i][0] == i % 2)rowSwap++;
        	if(board[0][i] == i % 2)colSwap++;
        }
        if(N / 2 > rowSum || rowSum > (N + 1) / 2)return -1;
        if(N / 2 > colSum || colSum > (N + 1) / 2)return -1;
        if(N % 2 == 1){
        	if(colSwap % 2 == 1)colSwap = N - colSwap;
        	if(rowSwap % 2 == 1)rowSwap = N - rowSwap;
        }else{
        	colSwap = Math.min(colSwap, N - colSwap);
        	rowSwap = Math.min(rowSwap, N - rowSwap);
        }
        return (rowSwap + colSwap) / 2;
    }

 * 2018/4/19
 * 730. Count Different Palindromic Subsequences
 * 
    public int countPalindromicSubsequences(String S) {
        if(S == null || S.length() == 0)return 0;
        int len = S.length();
        char[] ch = S.toCharArray();
        int[][] dp = new int[len][len];
        for(int i = 0;i < len;i++)dp[i][i] = 1;
        for(int distance = 1;distance < len;distance++){
        	for(int i = 0;i < len - distance;i++){
        		int j = i + distance;
        		if(ch[i] == ch[j]){
        			int low = i + 1;
        			int high = j - 1;
        			while(low <= high && ch[low] != ch[j])low++;
        			while(low <= high && ch[high] != ch[j])high--;
        			if(low > high){
        				dp[i][j] = dp[i + 1][j - 1] * 2 + 2;
        			}
        			else if(low == high){
        				dp[i][j] = dp[i + 1][j - 1] * 2 + 1;
        			}
        			else{
        				dp[i][j] = dp[i + 1][j - 1] * 2 - dp[low + 1][high - 1];
        			}
        		}else{
        			dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1];
        		}
        		dp[i][j] = dp[i][j] < 0 ? dp[i][j] + 1000000007 : dp[i][j] % 1000000007;
        	}
        }
        return dp[0][len - 1];
    }
    
 * 2018/4/19
 * 51. N-Queens
 * 
class Solution {
    public List<List<String>> solveNQueens(int n) {
        char[][] board = new char[n][n];
        for(int i = 0;i < n;i++){
        	for(int j = 0;j < n;j++)
        		board[i][j] = '.';
        }
        List<List<String>> res = new ArrayList<>();
        dfs(board, 0, res);
        return res;
    }
    
    public void dfs(char[][] board, int colIndex, List<List<String>> res){
    	if(colIndex == board.length){
    		res.add(construct(board));
    	}
    	
    	for(int i = 0;i < board.length;i++){
    		if(isValid(board, i, colIndex)){
    			board[i][colIndex] = 'Q';
    			dfs(board, colIndex + 1, res);
    			board[i][colIndex] = '.';
    		}
    	}
    }
    
    public boolean isValid(char[][] board, int x, int y){//(x,y)置为'Q'，判断是否成立
    	for(int i = 0;i < board.length;i++){
    		for(int j = 0;j < y;j++){
    			if(board[i][j] == 'Q' && (x + j == y + i || x + y == i + j || x == i)){
    				return false;
    			}
    		}
    	}
    	return true;
    }
    
    public List<String> construct(char[][] board){
        List<String> res = new ArrayList<>();
        for(int i = 0; i < board.length; i++) {
            String s = new String(board[i]);
            res.add(s);
        }
        return res;
    }
}
*/

/**
 * 2018/4/20
 * 691. Stickers to Spell Word
 * 
class Solution {
    public int minStickers(String[] stickers, String target) {
        int[] count = new int[26];
        for(char c : target.toCharArray())count[c - 'a']++;
        if(!isGetResult(stickers, count))return -1;
        return help(stickers, count);
    }
    
    //判断是否不能生成target
    public boolean isGetResult(String[] stickers, int[] count){
    	boolean[] valid = new boolean[26];
    	for(int i = 0;i < 26;i++){
    		if(count[i] > 0)valid[i] = true;
    	}
    	for(String sticker : stickers){
    		for(char ch : sticker.toCharArray()){
    			if(valid[ch - 'a'] == true)
    				valid[ch - 'a'] = false;
    		}
    	}
    	for(int i = 0;i < 26;i++){
    		if(valid[i])return false;
    	}
    	return true;
    }
    
    Map<String, Integer> map = new HashMap<>();
    
    public int help(String[] stickers, int[] count){
    	boolean flag = true;
    	for(int i = 0;i < 26;i++){
    		if(count[i] > 0){
    			flag = false;
    			break;
    		}
    	}
    	if(flag)return 0;
    	String code = encode(count);
    	if(map.containsKey(code))return map.get(code);
    	int min = Integer.MAX_VALUE;
    	for(String sticker : stickers){
    		if(isValid(sticker, count)){
    			int[] temp = new int[26];
    			for(int i = 0;i < 26;i++)temp[i] = count[i];
    			for(char ch : sticker.toCharArray()){
    				temp[ch - 'a']--;
    			}
    			min = Math.min(min, help(stickers, temp) + 1);
    		}
    	}
    	map.put(code, min);
    	return min;
    }
    
    public boolean isValid(String sticker, int[] count){
    	for(char ch : sticker.toCharArray()){
    		if(count[ch - 'a'] > 0)return true;
    	}
    	return false;
    }
    
    public String encode(int[] count){
    	StringBuffer res = new StringBuffer();
    	for(int i = 0;i < 26;i++){
    		if(count[i] > 0){
    			res.append((char)('a' + i)).append(count[i]);
    		}
    	}
    	return res.toString();
    }
}

 * 2018/4/20
 * 239. Sliding Window Maximum
 * 
    public int[] maxSlidingWindow(int[] nums, int k) {
    	if(nums == null || nums.length == 0)return new int[]{0};
        PriorityQueue<Integer> queue = new PriorityQueue<>((a, b) -> b - a);
        for(int i = 0;i < k;i++){
        	queue.add(nums[i]);
        }
        int len = nums.length;
        int[] res = new int[len - k + 1];
        res[0] = queue.peek();
        for(int i = k;i < len;i++){
        	queue.remove(nums[i - k]);
        	queue.add(nums[i]);
        	res[i - k + 1] = queue.peek();
        }
        return res;
    }
    
 * 2018/4/20
 * 818. Race Car
 * 
    public int racecar(int target) {
    	int[] dp = new int[target + 3];
    	Arrays.fill(dp, Integer.MAX_VALUE);
    	dp[0] = 0; dp[1] = 1; dp[2] = 4;
    	for(int t = 3;t <= target;t++){
    		int k = 32 - Integer.numberOfLeadingZeros(t);
    		if((1 << k) - 1 == t){
    			dp[t] = k;
    			continue;
    		}
    		for(int j = 0;j < k - 1;j++){
    			dp[t] = Math.min(dp[t], (k - 1) + j + 2 + dp[t - (1 << (k - 1)) + (1 << j)]);
    		}
    		if((1 << k) - 1 - t < t){
    			dp[t] = Math.min(dp[t], k + 1 + dp[(1 << k) - 1 - t]);
    		}
    	}
    	return dp[target];
    }
    
 * 2018/4/20
 * 664. Strange Printer
 * 
class Solution {
    public int strangePrinter(String s) {
    	if(s == null || s.length() == 0)return 0;
        StringBuffer temp = new StringBuffer();
        temp.append(s.charAt(0));
        for(int i = 1;i < s.length();i++){
        	if(s.charAt(i) != s.charAt(i - 1)){
        		temp.append(s.charAt(i));
        	}
        }
        int[][] dp = new int[temp.length()][temp.length()];
        for(int i = 0;i < temp.length();i++)dp[i][i] = 1;
        return help(temp.toString(), dp, 0, temp.length() - 1);
    }
    
    public int help(String str, int[][] dp, int l, int r){
    	if(l > r)return 0;
    	if(dp[l][r] != 0)return dp[l][r];
    	int res = 1 + help(str, dp, l + 1, r);
    	for(int i = l + 1;i <= r;i++){
    		if(str.charAt(i) == str.charAt(l)){
    			res = Math.min(res, help(str, dp, l, i - 1) + help(str, dp, i + 1, r));
    		}
    	}
    	dp[l][r] = res;
    	return res;
    }
}
*/

/**
 * 2018/4/21
 * 757. Set Intersection Size At Least Two
 * 
    public int intersectionSizeTwo(int[][] intervals) {
    	if(intervals == null || intervals.length == 0)return 0;
        Arrays.sort(intervals, new Comparator<int[]>() {
			public int compare(int[] o1, int[] o2) {
				if(o1[0] == o2[0]){
					return o2[1] - o1[1];
				}
				return o1[0] - o2[0];
			}
		});
        int res = 0;
        int[] total = new int[intervals.length];
        Arrays.fill(total, 2);
        int t = intervals.length;
        while(--t >= 0){
        	int start = intervals[t][0];
        	int m = total[t];
        	for(int pos = start;pos < start + m;pos++){
        		for(int i = 0;i <= t;i++){
        			if(total[i] > 0 && pos <= intervals[i][1]){
        				total[i]--;
        			}
        		}
        		res++;
        	}
        }
        return res;
    }
*/

/**
 * 2018/4/22
 * 483. Smallest Good Base
 * 
    public String smallestGoodBase(String n) {
        long num = Long.parseLong(n);
        long res = 0;
        for(int k = 60;k >= 2;k--){
        	long start = 2;
        	long end = num;
        	while(start < end){
        		long mid = (start + end) / 2;
        		BigInteger left = BigInteger.valueOf(mid).pow(k).subtract(BigInteger.ONE);
        		BigInteger right = BigInteger.valueOf(num).multiply(BigInteger.valueOf(mid).subtract(BigInteger.ONE));
        		int cmp = left.compareTo(right);
        		if(cmp == 0){
        			res = mid;
        			break;
        		}else if(cmp < 0){
        			start = mid + 1;
        		}else{
        			end = mid;
        		}
        	}
        	if(res != 0)break;
        }
        return "" + res;
    }
    
 * 2018/4/22
 * 821. Shortest Distance to a Character
 * 
    public int[] shortestToChar(String S, char C) {
    	if(S == null || S.length() == 0)return new int[0];
    	List<Integer> list = new ArrayList<>();//index of C in S
        list.add(30000);
        for(int i = 0;i < S.length();i++){
        	if(S.charAt(i) == C)list.add(i);
        }
        list.add(30000);
        int[] res = new int[S.length()];
        for(int i = 0;i < S.length();i++){
        	if(S.charAt(i) == C){
        		res[i] = 0;
        		list.remove(0);
        	}
        	else{
        		res[i] = Math.min(Math.abs(i - list.get(0)), Math.abs(i - list.get(1)));
        	}
        }
        return res;
    }
    
 * 2018/4/22
 * 822. Card Flipping Game
 * 
class Solution {
    public int flipgame(int[] fronts, int[] backs) {
        Set<Integer> depositAccessedNum = new HashSet<>();
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        for(int front : fronts){
        	if(!depositAccessedNum.contains(front)){
        		depositAccessedNum.add(front);
        		queue.add(front);
        	}
        }
        for(int back : backs){
        	if(!depositAccessedNum.contains(back)){
        		depositAccessedNum.add(back);
        		queue.add(back);
        	}
        }
        while(!queue.isEmpty()){
        	int num = queue.poll();
        	if(isValid(fronts, backs, num))return num;
        }
        return 0;
    }
    
    public boolean isValid(int[] fronts, int[] backs, int num){
    	for(int i = 0;i < fronts.length;i++){
    		if(num == fronts[i] && num == backs[i]){
    			return false;
    		}
    	}
    	return true;
    }
}

 * 2018/4/22
 * 820. Short Encoding of Words
 * 
class Solution {
    public int minimumLengthEncoding(String[] words) {
        Arrays.sort(words, new Comparator<String>() {
			public int compare(String o1, String o2) {
				if(o1.length() == o2.length())return o1.compareTo(o2);
				return o2.length() - o1.length();
			}
		});
        int res = 0;
        boolean[] visit = new boolean[words.length];
        for(int i = 0;i < words.length;i++){
        	if(!visit[i]){
        		String cur = words[i];
        		for(int j = i + 1;j < words.length;j++){
        			if(!visit[j]){
        				if(isValid(cur, words[j])){
        					visit[j] = true;
        				}
        			}
        		}
        		res += cur.length() + 1;
        	}
        }
        return res;
    }
    
    public boolean isValid(String str1, String str2){
    	int i = str1.length() - 1;
    	int j = str2.length() - 1;
    	while(j >= 0){
    		if(str2.charAt(j--) != str1.charAt(i--))return false;
    	}
    	return true;
    }
}

 * 2018/4/22
 * 823. Binary Trees With Factors
 * 
    public int numFactoredBinaryTrees(int[] A) {
        Arrays.sort(A);
        long[] dp = new long[A.length];
        for(int i = 0;i < A.length;i++){
        	long resI = 1;
			for (int j = 0; j < i; j++) {
				if (A[i] % A[j] == 0) {
					for (int k = 0; k < i; k++) {
						if (A[i] == A[j] * A[k]) {
							resI += (dp[j] * dp[k]) % 1000000007;
							resI %= 1000000007;
						}else if(A[j] * A[k] > A[i])break;
					}
				}
        	}
        	dp[i] = resI;
        }
        long res = 0;
        for(long d : dp){
        	res += d;
        	res %= 1000000007;
        }
        return (int)res;
    }
    
 * 2018/4/22
 * 363. Max Sum of Rectangle No Larger Than K
 * 
    public int maxSumSubmatrix(int[][] matrix, int k) {
        if(matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)return 0;
        int row = matrix.length;
        int col = matrix[0].length;
        int[][] areas = new int[row][col];
        for(int i = 0;i < row;i++){
        	for(int j = 0;j < col;j++){
        		int area = matrix[i][j];
        		if(i - 1 >= 0) area += areas[i - 1][j];
        		if(j - 1 >= 0) area += areas[i][j - 1];
        		if(i - 1 >= 0 && j - 1 >= 0) area -= areas[i - 1][j - 1];
        		areas[i][j] = area;
        	}
        }
        int max = Integer.MIN_VALUE;
        for(int r1 = 0;r1 < row;r1++){
        	for(int r2 = r1;r2 < row;r2++){
        		TreeSet<Integer> tree = new TreeSet<>();
        		tree.add(0);
        		for(int c = 0;c < col;c++){
        			int area = areas[r2][c];
        			if(r1 - 1 >= 0)area -= areas[r1 - 1][c];
        			Integer celling = tree.ceiling(area - k);
        			if(celling != null){
        				max = Math.max(max, area - celling);
        			}
        			tree.add(area);
        		}
        	}
        }
        return max;
    }
    
 * 2018/4/22
 * 587. Erect the Fence
 * 
class Solution {
	class Point {
		int x;
		int y;
		Point() {x = 0; y = 0;}
		Point(int a, int b) {x = a; y = b;}
	}
	
	public int orientation(Point p, Point q, Point r){
		return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
	}
	
	public boolean isBetween(Point p, Point i, Point q){
        boolean a = i.x >= p.x && i.x <= q.x || i.x <= p.x && i.x >= q.x;
        boolean b = i.y >= p.y && i.y <= q.y || i.y <= p.y && i.y >= q.y;
        return a && b;
	}
	
    public List<Point> outerTrees(Point[] points) {
        HashSet<Point> hull = new HashSet<>();
        if(points.length < 4){
        	for(Point point : points)hull.add(point);
        	return new ArrayList<Point>(hull);
        }
        int left_most = 0;
        for(int i = 1;i < points.length;i++){
        	if(points[i].x < points[left_most].x){
        		left_most = i;
        	}
        }
        int p = left_most;
        do{
        	int q = (p + 1) % points.length;
        	for(int i = 0;i < points.length;i++){
        		if(orientation(points[p], points[i], points[q]) < 0){
        			q = i;
        		}
        	}
        	for(int i = 0;i < points.length;i++){
        		if(i != p && i != q && orientation(points[p], points[i], points[q]) == 0 && isBetween(points[p], points[i], points[q])){
        			hull.add(points[i]);
        		}
        	}
        	hull.add(points[q]);
        	p = q;
        }while(p != left_most);
        return new ArrayList<Point>(hull);
    }
}
*/

/**
 * 2018/4/23
 * 149. Max Points on a Line
 * 
class Solution {
	class Point {
		int x;
		int y;
		Point() { x = 0; y = 0; }
		Point(int a, int b) { x = a; y = b; }
	}
    public int maxPoints(Point[] points) {
        if(points == null || points.length == 0)return 0;
        if(points.length <= 2)return points.length;
        Map<Integer, Map<Integer,Integer>> map = new HashMap<>();
        int result = 0;
        for(int i = 0;i < points.length;i++){
        	map.clear();
        	Point point = points[i];
        	int overlap = 0;
        	int max = 0;
        	for(int j = i + 1;j < points.length;j++){
        		Point p = points[j];
        		int x = p.x - point.x;
        		int y = p.y - point.y;
        		if(x == 0 && y == 0){
        			overlap++;
        			continue;
        		}
        		int gcd = GCD(x, y);
        		if(gcd != 0){
        			x /= gcd;
        			y /= gcd;
        		}
        		if(map.containsKey(x)){
        			map.get(x).put(y, map.get(x).getOrDefault(y, 0) + 1);
    			}else{
    				Map<Integer,Integer> m = new HashMap<Integer,Integer>();
    				m.put(y, 1);
    				map.put(x, m);
    			}
        		max = Math.max(max, map.get(x).get(y));
        	}
        	result = Math.max(result, max + overlap + 1);
        }
        return result;
    }
    
    public int GCD(int a, int b){
    	if(b == 0)return a;
    	return GCD(b, a % b);
    }
}

 * 2018/4/23
 * 330. Patching Array
 * 
    public int minPatches(int[] nums, int n) {
    	long miss = 1;
    	int add = 0;
    	int i = 0;
    	while(miss <= n){
    		if(i < nums.length && nums[i] <= miss){
    			miss += nums[i++];
    		}else{
    			miss += miss;
    			add++;
    		}
    	}
    	return add;
    }
*/

/**
 * 2018/4/24
 * 354. Russian Doll Envelopes
 * 
    public int maxEnvelopes(int[][] envelopes) {
    	if(envelopes == null || envelopes.length == 0)return 0;
        Arrays.sort(envelopes, (a, b) -> (a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]));
        int[] dp = new int[envelopes.length];
        for(int i = 0;i < dp.length;i++){
        	dp[i] = 1;
        	for(int j = i - 1;j >= 0;j--){
        		if(envelopes[i][0] > envelopes[j][0] && envelopes[i][1] > envelopes[j][1]){
        			dp[i] = Math.max(dp[i], dp[j] + 1);
        		}
        	}
        }
        int res = Integer.MIN_VALUE;
        for(int i = 0;i < dp.length;i++){
        	res = Math.max(res, dp[i]);
        }
        return res;
    }
    
 * 2018/4/24
 * 403. Frog Jump
 * 
    public boolean canCross(int[] stones) {
    	if(stones[1] != 1)return false;
    	Map<Integer, Set<Integer>> map = new HashMap<>(stones.length);
    	map.put(0, new HashSet<Integer>());
        map.get(0).add(1);
        for (int i = 1; i < stones.length; i++) {
        	map.put(stones[i], new HashSet<Integer>() );
        }
        for(int i = 0;i < stones.length;i++){
        	for(int step : map.get(stones[i])){
        		int reach = step + stones[i];
        		if(reach == stones[stones.length - 1])return true;
        		Set<Integer> set = map.get(reach);
        		if(set != null){
        			set.add(step);
        			if(step - 1 > 0)set.add(step - 1);
        			set.add(step + 1);
        		}
        	}
        }
        return false;
    }
*/
package exercise;
