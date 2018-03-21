/**
 * @author jhy code from 3.14 to 
 *  questions
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
package exercise;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.TreeSet;

class Solution {
	
}







