/**
 * @author jhy code from 5.20 to 5.25
 * 23 questions
 */

/**
 * 2018/5/20
 * 97. Interleaving String
 * 
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s3.length() != s1.length() + s2.length()) {
            return false;
        }
        boolean dp[][] = new boolean[s1.length() + 1][s2.length() + 1];
        for (int i = 0; i <= s1.length(); i++) {
            for (int j = 0; j <= s2.length(); j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = true;
                } else if (i == 0) {
                    dp[i][j] = dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1);
                } else if (j == 0) {
                    dp[i][j] = dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1);
                } else {
                    dp[i][j] = (dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1)) || (dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
                }
            }
        }
        return dp[s1.length()][s2.length()];
    }

 * 2018/5/20
 * 836. Rectangle Overlap
 * 
    public boolean isRectangleOverlap(int[] rec1, int[] rec2) {
        boolean rowOverlap = Math.max(rec1[0], rec2[0]) < Math.min(rec1[2], rec2[2]);
        boolean colOverlap = Math.max(rec1[1], rec2[1]) < Math.min(rec1[3], rec2[3]);
        return rowOverlap && colOverlap;
    }

 * 2018/5/20
 * 837. New 21 Game
 * 
    public double new21Game(int N, int K, int W) {
    	if(K == 0)return 1.0;
        double[] dp = new double[N + 1];
        dp[0] = 1.0;
        double sum = 1.0;
        double res = 0;
        for(int i = 1;i <= N;i++){
        	dp[i] = sum / W;
        	if(i < K)sum += dp[i];
        	else res += dp[i];
        	if(i - W >= 0 && i - W < K)sum -= dp[i - W];
        }
        return res;
    }
    
 * 2018/5/20
 * 838. Push Dominoes
 * 
    public String pushDominoes(String dominoes) {
        StringBuilder res = new StringBuilder();
        boolean preIsRight = false;
        int pre = 0;
        for(int i = 0;i < dominoes.length();i++){
        	if(dominoes.charAt(i) == 'L'){
        		if(preIsRight){//之前有R，pre i 间R L 参半
        			int num = (i - pre + 1) / 2;
        			boolean isEven = (i - pre) % 2 == 1;
        			for(int j = 0;j < num;j++)res.append("R");
        			if(!isEven)res.append(".");
        			for(int j = 0;j < num;j++)res.append("L");
        		}else{
        			for(int j = pre;j <= i;j++){
        				res.append("L");
        			}
        		}
        		pre = i + 1;
        		preIsRight = false;
        	}else if(dominoes.charAt(i) == 'R'){
        		if(preIsRight){
        			for(int j = pre;j < i;j++)res.append("R");
        		}else{
        			for(int j = pre;j < i;j++)res.append(".");
        		}
        		pre = i;
        		preIsRight = true;
        	}else{
        		continue;
        	}
        }
        if(pre == dominoes.length())return res.toString();
        if(dominoes.charAt(pre) == 'R'){
        	for(int j = pre;j < dominoes.length();j++){
        		res.append("R");
        	}
        }else{
        	for(int j = pre;j < dominoes.length();j++){
        		res.append(".");
        	}
        }
        return res.toString();
    }
    
 * 2018/5/20
 * 839. Similar String Groups
 * 
class Solution {
    public int numSimilarGroups(String[] A) {
        if(A.length < 2)return A.length;
        int res = 0;
        for(int i = 0;i < A.length;i++){
        	if(A[i] == null)continue;
        	String str = A[i];
        	A[i] = null;
        	res++;
        	dfs(A, str);
        }
        return res;
    }
    
    public void dfs(String[] A, String str){
    	for(int i = 0;i < A.length;i++){
    		if(A[i] == null)continue;
    		if(helper(A[i], str)){
    			String s = A[i];
    			A[i] = null;
    			dfs(A, s);
    		}
    	}
    }
    
    public boolean helper(String s, String t){
    	int res = 0;
    	int i = 0;
    	while(res <= 2 && i < s.length()){
    		if(s.charAt(i) != t.charAt(i))res++;
    		i++;
    	}
    	return res == 2;
    }
}
*/

/**
 * 2018/5/21
 * 214. Shortest Palindrome
 * 
    public String shortestPalindrome(String s) {//KMP algorithm
    	String rever = new StringBuilder(s).reverse().toString();
    	String strNew = s + "#" + rever;
    	int[] KMP = new int[strNew.length()];
    	for(int i = 1;i < KMP.length;i++){
    		int temp = KMP[i - 1];
    		while(temp > 0 && strNew.charAt(temp) != strNew.charAt(i)){
    			temp = KMP[temp - 1];
    		}
    		if(strNew.charAt(temp) == strNew.charAt(i)){
    			temp++;
    		}
    		KMP[i] = temp;
    	}
    	return rever.substring(0, s.length() - KMP[KMP.length - 1]) + s;
    }
    
 * 2018/5/21
 * 135. Candy
 * 
    public int candy(int[] ratings) {
        if(ratings == null || ratings.length == 0)return 0;
        int res = 0;
    	int N = ratings.length;
        int[] leftToRight = new int[N];
        int[] rightToLeft = new int[N];
        leftToRight[0] = 1;
        for(int i = 1;i < N;i++){
        	if(ratings[i] > ratings[i - 1]){
        		leftToRight[i] = leftToRight[i - 1] + 1;
        	}else{
        		leftToRight[i] = 1;
        	}
        }
        rightToLeft[N - 1] = 1;
        for(int i = N - 2;i >= 0;i--){
        	if(ratings[i] > ratings[i + 1]){
        		rightToLeft[i] = rightToLeft[i + 1] + 1;
        	}else{
        		rightToLeft[i] = 1;
        	}
        }
        for(int i = 0;i < N;i++){
        	res += Math.max(leftToRight[i], rightToLeft[i]);
        }
        return res;
    }

 * 2018/5/21
 * 639. Decode Ways II
 * 
    public int numDecodings(String s) {
        long[] dp = new long[s.length() + 1];
        dp[0] = 1;
        if(s.charAt(0) == '0')return 0;
        dp[1] = (s.charAt(0) == '*') ? 9 : 1;
        for(int i = 2;i < dp.length;i++){
        	char first = s.charAt(i - 2);
        	char second = s.charAt(i - 1);
        	
        	if(second == '*'){
        		dp[i] += 9 * dp[i - 1];
        	}else if(second > '0'){
        		dp[i] += dp[i - 1];
        	}
        	
        	if(first == '*'){
                if(second == '*'){
                    dp[i] += 15*dp[i-2];
                }else if(second <= '6'){
                    dp[i] += 2*dp[i-2];
                }else{
                    dp[i] += dp[i-2];
                }
            }else if(first == '1' || first == '2'){
                if(second == '*'){
                    if(first == '1'){
                       dp[i] += 9*dp[i-2]; 
                    }else{ // first == '2'
                       dp[i] += 6*dp[i-2]; 
                    }
				} else if (((first - '0') * 10 + (second - '0')) <= 26) {
                    dp[i] += dp[i-2];    
                }
            }
            dp[i] %= 1000000007;
        }
        return (int)dp[s.length()];
    }
    
 * 2018/5/21
 * 132. Palindrome Partitioning II
 * 
    public int minCut(String s) {
        int n = s.length();
        int[] dp = new int[n + 1];//前i字符需要切的次数
        for(int i = 0;i < dp.length;i++)
        	dp[i] = i - 1;
        for(int i = 0;i < n;i++){
        	for(int j = 0;i - j >= 0 && i + j < n && s.charAt(i - j) == s.charAt(i + j);j++){
        		dp[i + j + 1] = Math.min(dp[i + j + 1], dp[i - j] + 1);
        	}
        	for(int j = 1;i - j + 1 >= 0 && i + j < n && s.charAt(i - j + 1) == s.charAt(i + j);j++){
        		dp[i + j + 1] = Math.min(dp[i + j + 1], dp[i - j + 1] + 1);
        	}
        }
        return dp[n];
    }
*/

/**
 * 2018/5/22
 * 321. Create Maximum Number
 * 
class Solution {
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int n = nums1.length;
        int m = nums2.length;
        int[] res = new int[k];
    	for(int i = Math.max(0, k - m);i <= n && i <= k;i++){
    		int[] temp = merge(maxArray(nums1, i), maxArray(nums2, k - i), k);
    		if(greater(temp, 0, res, 0))res = temp;
    	}
    	return res;
    }
    
    public int[] merge(int[] num1, int[] num2, int k){
    	int[] res = new int[k];
    	for(int i = 0, j = 0, r = 0;r < k;r++){
    		res[r] = greater(num1, i, num2, j) ? num1[i++] : num2[j++];
    	}
    	return res;
    }
    
    public boolean greater(int[] num1, int i, int[] num2, int j){
    	while(i < num1.length && j < num2.length && num1[i] == num2[j]){
    		i++;
    		j++;
    	}
    	return j == num2.length || (i < num1.length && num1[i] > num2[j]);
    }
    
    public int[] maxArray(int[] num, int k){
    	int n = num.length;
    	int[] res = new int[k];
    	for(int i = 0,j = 0;i < num.length;i++){
    		while(j + n - i > k && j > 0 && res[j - 1] < num[i])j--;
    		if(j < k)res[j++] = num[i];
    	}
    	return res;
    }
}
*/

/**
 * 2018/5/23
 * 30. Substring with Concatenation of All Words
 * 
    public List<Integer> findSubstring(String s, String[] words) {
        int len = s.length();
        int n = words.length;
        List<Integer> res = new ArrayList<>();
        if(len == 0 || n == 0)return res;
        int k = words[0].length();
        Map<String, Integer> map = new HashMap<>();
        for(String word : words)
        	map.put(word, map.getOrDefault(word, 0) + 1);
        for(int i = 0;i < k;i++){
        	Map<String, Integer> tempMap = new HashMap<>();
        	int count = 0;
        	int left = i;
        	for(int right = i;right + k <= len;right += k){
        		String str = s.substring(right, right + k);
        		if(!map.containsKey(str)){
        			left = right + k;
        			tempMap.clear();
        			count = 0;
        		}else{//str在words中
        			while(tempMap.containsKey(str) && tempMap.get(str) == map.get(str)){
        				String leftStr = s.substring(left, left + k);
        				tempMap.put(leftStr, tempMap.get(leftStr) - 1);
        				left += k;
        				count--;
        			}
        			tempMap.put(str, tempMap.getOrDefault(str, 0) + 1);
        			count++;
        			if(count == n){
        				res.add(left);
        			}
        		}
        	}
        }
        return res;
    }
    
 * 2018/5/23
 * 212. Word Search II
 * 
class Solution {//Trie + backtracing
	class TrieNode{
		public boolean isWord;
		public TrieNode[] trs = new TrieNode[26];
		public TrieNode(){}
	}
	
	public TrieNode root;
	
	public void insert(String str){
		if(str == null || str.length() == 0)return;
		TrieNode temp = root;
		for(char ch : str.toCharArray()){
			if(temp.trs[ch - 'a'] == null){
				temp.trs[ch - 'a'] = new TrieNode();
			}
			temp = temp.trs[ch - 'a'];
		}
		temp.isWord = true;
	}
	
	public boolean isWord(String str){
		if(str == null || str.length() == 0)return false;
		TrieNode temp = root;
		for(char ch : str.toCharArray()){
			if(temp.trs[ch - 'a'] == null)
				return false;
			temp = temp.trs[ch - 'a'];
		}
		return temp.isWord;
	}
	
	public boolean isPerfix(String str){
		if(str == null || str.length() == 0)return false;
		TrieNode temp = root;
		for(char ch : str.toCharArray()){
			if(temp.trs[ch - 'a'] == null)
				return false;
			temp = temp.trs[ch - 'a'];
		}
		return true;
	}
	
	int m;
	int n;
	int[][] dir = new int[][]{{0,1}, {0,-1}, {1,0}, {-1,0}};
	
    public List<String> findWords(char[][] board, String[] words) {
    	List<String> res = new ArrayList<>();
    	if(board == null || board.length == 0 || board[0] == null || board[0].length == 0)return res;
        root = new TrieNode();
        for(String word : words){
        	insert(word);
        }
        m = board.length;
        n = board[0].length;
        StringBuilder str = new StringBuilder();
        HashSet<String> setRes = new HashSet<>();
        boolean[][] visited = new boolean[m][n];
        for(int i = 0;i < m;i++){
        	for(int j = 0;j < n;j++){
        		str.append(board[i][j]);
        		backtracing(board, visited, i, j, str, setRes);
        		str.setLength(0);
        	}
        }
        return new ArrayList<String>(setRes);
    }
    
    public void backtracing(char[][] board, boolean[][] visited, int i, int j, StringBuilder str, Set<String> set){
    	String toString = str.toString();
    	if(!isPerfix(toString))return;
    	if(isWord(toString)){
    		set.add(toString);
    	}
    	visited[i][j] = true;
    	for(int[] d : dir){
    		int row = i + d[0];
    		int col = j + d[1];
    		if(row >= 0 && row < m && col >= 0 && col < n && !visited[row][col]){
    			str.append(board[row][col]);
    			backtracing(board, visited, row, col, str, set);
    			str.setLength(str.length() - 1);
    		}
    	}
    	visited[i][j] = false;
    }
}

 * 2018/5/23
 * 44. Wildcard Matching
 * 
    public boolean isMatch(String s, String p) {
        int sIndex = 0;
        int pIndex = 0;
        int startIndex = -1;
        int match = 0;
        while(sIndex < s.length()){
        	if(pIndex < p.length() && (p.charAt(pIndex) == '?' || s.charAt(sIndex) == p.charAt(pIndex))){
        		sIndex++;
        		pIndex++;
        	}else if(pIndex < p.length() && p.charAt(pIndex) == '*'){
        		startIndex = pIndex;
        		match = sIndex;
        		pIndex++;
        	}else if(startIndex != -1){
        		pIndex = startIndex + 1;
        		match++;
        		sIndex = match;
        	}else{
        		return false;
        	}
        }
        while(pIndex < p.length() && p.charAt(pIndex) == '*')
        	pIndex++;
        return pIndex == p.length();
    }
*/

/**
 * 2018/5/24
 * 68. Text Justification
 * 
    public List<String> fullJustify(String[] words, int maxWidth) {
    	List<String> res = new ArrayList<>();
    	if(words == null || words.length == 0)return res;
    	int left = 0;
    	while(left < words.length){
    		StringBuilder temp = new StringBuilder();
    		int right = left;
    		int wordLen = 0;
    		int totalLen = 0;
    		int count = 0;
    		while(right < words.length && totalLen + words[right].length() <= maxWidth){
    			wordLen += words[right].length();
    			totalLen += words[right].length() + 1;
    			count++;
    			right++;
    		}
    		if(right == words.length){//到达最后一行
    			temp.append(words[left]);
    			for(int i = left + 1;i < right;i++){
    				temp.append(" ").append(words[i]);
    			}
    			while(temp.length() < maxWidth){
    				temp.append(" ");
    			}
    			res.add(temp.toString());
    			return res;
    		}else{//未到最后一行
    			temp.append(words[left]);
    			if(count == 1){
    				while(temp.length() < maxWidth){
        				temp.append(" ");
        			}
    			}else{
    				count--;
    				int space = maxWidth - wordLen;
    				int preCount = space % count;
    				int postCount = count - preCount;
    				int preNum = space / count + 1;
    				int postNum = preNum - 1;
    				int index = left + 1;
    				for(int i = 0;i < preCount;i++){
    					for(int j = 0;j < preNum;j++){
    						temp.append(" ");
    					}
    					temp.append(words[index++]);
    				}
    				for(int i = 0;i < postCount;i++){
    					for(int j = 0;j < postNum;j++){
    						temp.append(" ");
    					}
    					temp.append(words[index++]);
    				}
    			}
    			res.add(temp.toString());
    		}
    		left = right;
    	}
    	return res;
    }
    
 * 2018/5/24
 * 10. Regular Expression Matching
 * 
class Solution {
    enum RESULT{
    	TRUE,FALSE;
    }
	
	public boolean isMatch(String s, String p) {
        RESULT[][] dp = new RESULT[s.length() + 1][p.length() + 1];
        return DP(0, 0, s, p, dp);
    }
	
	public boolean DP(int i, int j, String s, String p, RESULT[][] dp){
		if(dp[i][j] != null){
			return dp[i][j] == RESULT.TRUE ? true : false;
		}
		boolean res = false;
		if(j == p.length()){
			res = i == s.length();
		}else{
			boolean first = false;
			if(i < s.length() && (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.')){
				first = true;
			}
			if(j + 1 < p.length() && p.charAt(j + 1) == '*'){
				res = DP(i, j + 2, s, p, dp) || (first && DP(i + 1, j, s, p, dp));
			}else{
				res = first && DP(i + 1, j + 1, s, p, dp);
			}
		}
		dp[i][j] = res ? RESULT.TRUE : RESULT.FALSE;
		return res;
	}
}

 * 2018/5/24
 * 140. Word Break II
 * 
class Solution {
    public List<String> wordBreak(String s, List<String> wordDict) {
    	return dfs(s, wordDict, new HashMap<>());
    }
    
    public List<String> dfs(String s, List<String> wordDict, HashMap<String, LinkedList<String>> map){
    	if(map.containsKey(s)){
    		return map.get(s);
    	}
    	LinkedList<String> res = new LinkedList<>();
    	if(s == null || s.length() == 0){
    		res.add("");
    		return res;
    	}
    	for(String word : wordDict){
    		if(s.startsWith(word)){
    			List<String> lists = dfs(s.substring(word.length()), wordDict, map);
    			for(String list : lists){
    				res.add(word + (list.isEmpty() ? "" : " ") + list);
    			}
    		}
    	}
    	map.put(s, res);
    	return res;
    }
}

 * 2018/5/24
 * 440. K-th Smallest in Lexicographical Order
 * 
class Solution {
    public int findKthNumber(int n, int k) {
        int cur = 1;
        k--;
        while(k > 0){
        	int step = stepCalculate(n, cur, cur + 1);
        	if(step > k){
        		cur *= 10;
        		k--;
        	}else{
        		cur = cur + 1;
        		k -= step;
        	}
        }
        return cur;
    }
    
    public int stepCalculate(int n, long n1, long n2){//use long in case of overflow
    	int res = 0;
    	while(n1 <= n){
    		res += Math.min(n + 1, n2) - n1;
    		n1 *= 10;
    		n2 *= 10;
    	}
    	return res;
    }
}

 * 2018/5/24
 * 685. Redundant Connection II
 * 
class Solution {
    public int[] findRedundantDirectedConnection(int[][] edges) {
        int[] candiate1 = new int[]{-1, -1};
        int[] candiate2 = new int[]{-1, -1};
        int[] parent = new int[edges.length + 1];
        for(int[] edge : edges){
        	if(parent[edge[1]] != 0){
        		candiate1 = new int[]{parent[edge[1]], edge[1]};
        		candiate2 = new int[]{edge[0], edge[1]};
        		edge[0] = 0;
        	}else{
        		parent[edge[1]] = edge[0];
        	}
        }
        for(int i = 0;i < parent.length;i++){
        	parent[i] = i;
        }
        for(int[] edge : edges){
        	if(edge[0] == 0)continue;
        	int father = edge[0];
        	int child = edge[1];
        	if(root(parent, father) == child){
        		if(candiate1[0] != -1)
        			return candiate1;
        		return edge;
        	}
        	parent[child] = father;
        }
        return candiate2;
    }
    
    public int root(int[] parent, int i){
    	while(i != parent[i]){
    		parent[i] = parent[parent[i]];
    		i = parent[i];
    	}
    	return i;
    }
}

 * 2018/5/24
 * 174. Dungeon Game
 * 
    public int calculateMinimumHP(int[][] dungeon) {
        int m = dungeon.length;
        int n = dungeon[0].length;
        int[][] dp = new int[m + 1][n + 1];
        for(int i = 0;i < m + 1;i++){
        	Arrays.fill(dp[i], Integer.MAX_VALUE);
        }
        dp[m][n - 1] = 1;
        dp[m - 1][n] = 1;
        for(int i = m - 1;i >= 0;i--){
        	for(int j = n - 1;j >= 0;j--){
        		int need = Math.min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j];
        		dp[i][j] = need <= 0 ? 1 : need;
        	}
        }
        return dp[0][0];
    }
    
 * 2018/5/24
 * 420. Strong Password Checker
 * 
public class Solution {
    public int strongPasswordChecker(String s) {
        
        char [] str = s.toCharArray();
        boolean isUpper = false, isLower = false, isDigit = false;
        int missinType = 3;
        for(char c: str)
        {
            if(!isUpper && Character.isUpperCase(c)) { isUpper = true; missinType-=1; } //uppercase
            if(!isLower && Character.isLowerCase(c)) { isLower = true; missinType-=1; } //lowercase
            if(!isDigit && Character.isDigit(c)) { isDigit = true; missinType-=1; } //atleast one number
            
        }
       
        int totalChangeCnt = 0, OneChangeCnt =0, TwoChangeCnt =0, pos=2;
        while(pos < s.length())
        {
            if(str[pos]==str[pos-1] && str[pos-1]==str[pos-2] && str[pos-2]==str[pos])
            {
                int length = 2;
                while(pos < s.length() && str[pos]==str[pos-1])
                {
                    length += 1; pos +=1;
                }
                totalChangeCnt += length/3;
                if(length%3==0) OneChangeCnt += 1;
                else if(length%3==1) TwoChangeCnt += 1;
                
            }
            else
            {
                pos=pos+1;
            }
        }
        
        if(s.length()<6)
            return Math.max(missinType, 6-s.length());
        else if(s.length() <=20)
            return Math.max(missinType,totalChangeCnt );
        else
        {
            int deleteCount = s.length()-20;
            totalChangeCnt -= Math.min(deleteCount,OneChangeCnt*1)/1;
            totalChangeCnt -= Math.min(Math.max(deleteCount - OneChangeCnt, 0), TwoChangeCnt * 2) / 2;
            totalChangeCnt -= Math.max(deleteCount - OneChangeCnt - 2 * TwoChangeCnt, 0) / 3;
            
                
            return deleteCount + Math.max(missinType, totalChangeCnt);
        }       
    }
}
*/

/**
 * 2018/5/25
 * 126. Word Ladder II
 * 
class Solution {
	public List<List<String>> findLadders(String start, String end, List<String> wordList) {
	   HashSet<String> dict = new HashSet<String>(wordList);
	   List<List<String>> res = new ArrayList<List<String>>();         
	   HashMap<String, ArrayList<String>> nodeNeighbors = new HashMap<String, ArrayList<String>>();// Neighbors for every node
	   HashMap<String, Integer> distance = new HashMap<String, Integer>();// Distance of every node from the start node
	   ArrayList<String> solution = new ArrayList<String>();

	   dict.add(start);          
	   bfs(start, end, dict, nodeNeighbors, distance);                 
	   dfs(start, end, dict, nodeNeighbors, distance, solution, res);   
	   return res;
	}

	// BFS: Trace every node's distance from the start node (level by level).
	private void bfs(String start, String end, Set<String> dict, HashMap<String, ArrayList<String>> nodeNeighbors, HashMap<String, Integer> distance) {
	  for (String str : dict)
	      nodeNeighbors.put(str, new ArrayList<String>());

	  Queue<String> queue = new LinkedList<String>();
	  queue.offer(start);
	  distance.put(start, 0);

	  while (!queue.isEmpty()) {
	      int count = queue.size();
	      boolean foundEnd = false;
	      for (int i = 0; i < count; i++) {
	          String cur = queue.poll();
	          int curDistance = distance.get(cur);                
	          ArrayList<String> neighbors = getNeighbors(cur, dict);

	          for (String neighbor : neighbors) {
	              nodeNeighbors.get(cur).add(neighbor);
	              if (!distance.containsKey(neighbor)) {// Check if visited
	                  distance.put(neighbor, curDistance + 1);
	                  if (end.equals(neighbor))// Found the shortest path
	                      foundEnd = true;
	                  else
	                      queue.offer(neighbor);
	                  }
	              }
	          }

	          if (foundEnd)
	              break;
	      }
	  }

	// Find all next level nodes.    
	private ArrayList<String> getNeighbors(String node, Set<String> dict) {
	  ArrayList<String> res = new ArrayList<String>();
	  char chs[] = node.toCharArray();

	  for (char ch ='a'; ch <= 'z'; ch++) {
	      for (int i = 0; i < chs.length; i++) {
	          if (chs[i] == ch) continue;
	          char old_ch = chs[i];
	          chs[i] = ch;
	          if (dict.contains(String.valueOf(chs))) {
	              res.add(String.valueOf(chs));
	          }
	          chs[i] = old_ch;
	      }
	  }
	  return res;
	}

	// DFS: output all paths with the shortest distance.
	private void dfs(String cur, String end, Set<String> dict, HashMap<String, ArrayList<String>> nodeNeighbors, HashMap<String, Integer> distance, ArrayList<String> solution, List<List<String>> res) {
	    solution.add(cur);
	    if (end.equals(cur)) {
	       res.add(new ArrayList<String>(solution));
	    } else {
	       for (String next : nodeNeighbors.get(cur)) {
	            if (distance.get(next) == distance.get(cur) + 1) {
	                 dfs(next, end, dict, nodeNeighbors, distance, solution, res);
	            }
	        }
	    }
	   solution.remove(solution.size() - 1);
	}
}

 * 2018/5/25
 * 803. Bricks Falling When Hit
 * 
class Solution {
    public int[] hitBricks(int[][] grid, int[][] hits) {
        if(hits == null || hits.length == 0)return null;
        removeHits(grid, hits);
        markRemainBricks(grid);
        return searchFallingBrick(grid, hits);
    }
    
    public void removeHits(int[][] grid, int[][] hits){
    	for(int[] hit : hits){
    		grid[hit[0]][hit[1]] = grid[hit[0]][hit[1]] - 1;//如果原来是0，则变为-1
    	}
    }
    
    public void markRemainBricks(int[][] grid){
    	for(int i = 0;i < grid[0].length;i++){
    		deepSearch(grid, 0, i);
    	}
    }
    
    public int deepSearch(int[][] grid, int i, int j){
    	int row = grid.length;
    	int col = grid[0].length;
    	int res = 0;
    	if(i < 0 || i >= row || j < 0 || j >= col)return res;
    	if(grid[i][j] == 1){
    		grid[i][j] = 2;
    		res = 1;
    		res += deepSearch(grid, i + 1, j);
    		res += deepSearch(grid, i, j + 1);
    		res += deepSearch(grid, i - 1, j);
    		res += deepSearch(grid, i, j - 1);
    	}
    	return res;
    }
    
    public boolean isConnectToTop(int[][] grid, int i, int j) {
        if(i == 0) return true;
        if (i - 1 >= 0 && grid[i - 1][j] == 2) {
            return true;
        }
        if (i + 1 < grid.length && grid[i + 1][j] == 2) {
            return true;
        }
        if (j - 1 >= 0 && grid[i][j - 1] == 2) {
            return true;
        }
        if (j + 1 < grid[0].length && grid[i][j + 1] == 2) {
            return true;
        }
        return false;
    }
    
    public int[] searchFallingBrick(int[][] grid, int[][] hits){
    	int[] res = new int[hits.length];
    	for(int i = hits.length - 1;i >= 0;i--){
    		if(grid[hits[i][0]][hits[i][1]] == 0){
    			grid[hits[i][0]][hits[i][1]] = 1;
    			if(isConnectToTop(grid, hits[i][0], hits[i][1])){
    				res[i] = deepSearch(grid, hits[i][0], hits[i][1]) - 1;
    			}else{
    				res[i] = 0;
    			}
    		}
    	}
    	return res;
    }
}

 * 2018/5/25
 * 218. The Skyline Problem
 * 
    public List<int[]> getSkyline(int[][] buildings) {
    	List<int[]> result = new ArrayList<>();
    	List<int[]> height = new ArrayList<>();
    	for(int[] building : buildings){
    		height.add(new int[]{building[0], -building[2]});
    		height.add(new int[]{building[1], building[2]});
    	}
    	Collections.sort(height, (a, b)->{
    		if(a[0] == b[0]){
    			return a[1] - b[1];
    		}
    		return a[0] - b[0];
    	});
    	PriorityQueue<Integer> queue = new PriorityQueue<>((a, b) -> b - a);
    	int pre = 0;
    	queue.add(0);
    	for(int[] hei : height){
    		if(hei[1] < 0){
    			queue.add(-hei[1]);
    		}else{
    			queue.remove(hei[1]);
    		}
    		int cur = queue.peek();
    		if(cur != pre){
    			result.add(new int[]{hei[0], cur});
    			pre = cur;
    		}
    	}
    	return result;
    }
*/
package exercise;





