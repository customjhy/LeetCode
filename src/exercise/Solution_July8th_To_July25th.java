/**
 * @author jhy
 * code from 7.8 to 7.25
 * 36 questions
 */


/**
 * 2017/7/8
 * 507. Perfect Number
 * 
    public boolean checkPerfectNumber(int num) {
        if(num <= 0)return false;
        List<Integer> divisor = new ArrayList<Integer>();
        for(int i = 1;i <= (int)Math.sqrt(num);i++){
        	if(num % i == 0){
        		divisor.add(i);
        		divisor.add(num / i);
        	}
        }
        if((int)Math.sqrt(num) * (int)Math.sqrt(num) == num){
        	divisor.remove(divisor.size() - 1);
        }
        int sum = 0;
        for(int divi : divisor){
        	sum += divi;
        }
        if(sum == num * 2)return true;
        return false;
    }
*/

/**
 * 2017/7/9
 * 205. Isomorphic Strings
 * 
    public boolean isIsomorphic(String s, String t) {
        Map<Character,Character> temp = new HashMap<Character,Character>();
        for(int i = 0;i < s.length();i++){
        	if(temp.containsKey(s.charAt(i))){
        		if(temp.get(s.charAt(i)) != t.charAt(i))return false;
        	}
        	else{
        		for(int j = 0;j < i;j++){
        			if(t.charAt(i) == t.charAt(j))
        				return false;
        		}
        		temp.put(s.charAt(i), t.charAt(i));
        	}
        }
        return true;
    }
    
    
 * 2017/7/9
 * 20. Valid Parentheses
 * 
    public boolean isValid(String s) {
        Stack<Character> temp = new Stack<Character>();
        for(int i = 0;i < s.length();i++){
        	if(s.charAt(i) == '(' || s.charAt(i) == '[' || s.charAt(i) == '{'){
        		temp.push(s.charAt(i));
        	}
        	else{
        		if(temp.isEmpty()){
        			return false;
        		}
        		else{
        			if(!isMatched(temp.pop(),s.charAt(i))){
        				return false;
        			}
        		}
        	}
        }
        if(!temp.empty())
        	return false;
        return true;
    }
    
    public boolean isMatched(char a,char b){
    	if((a == '('&&b == ')') || (a == '[' && b == ']') || (a == '{' && b == '}'))
    		return true;
    	return false;
    }
    
    
 * 2017/7/9
 * 111. Minimum Depth of Binary Tree
 * 
    public int minDepth(TreeNode root) {
        if(root == null)return 0;
        if(root.left != null && root.right != null){
        	int left = minDepth(root.left);
        	int right = minDepth(root.right);
        	return (left < right ? left : right) + 1;
        }
        else if(root.left != null && root.right == null){
        	return minDepth(root.left) + 1;
        }
        else if(root.left == null && root.right != null){
        	return minDepth(root.right) + 1;
        }
        else{
        	return 1;
        }
    }
    
    
 * 2017/7/9
 * 290. Word Pattern
 * 
    public boolean wordPattern(String pattern, String str) {
        String[] temp  = str.split(" ");
        Map<Character,String> map = new HashMap<Character,String>();
        if(pattern.length() != temp.length)return false;
        for(int i = 0;i < pattern.length();i++){
        	if(!map.containsKey(pattern.charAt(i))){
        		if(map.containsValue(temp[i]))return false;
        		map.put(pattern.charAt(i), temp[i]);
        	}
        	else{
        		if(!temp[i].equals(map.get(pattern.charAt(i)))){
        			return false;
        		}
        	}
        }
        return true;
    }
*/

/**
 * 2017/7/10
 * 438. Find All Anagrams in a String
 * 
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> result = new ArrayList<Integer>();
        if(s.length() == 0||p.length() == 0||s == null||p == null){
        	return result;
        }
        int[] hash = new int[255];
        for(int i = 0;i < p.length();i++){
        	hash[p.charAt(i)]++;
        }
        int left = 0,right = 0;
        int count = p.length();
        while(right < s.length()){
        	if(hash[s.charAt(right)] >= 1){
        		count--;
        	}
        	hash[s.charAt(right)]--;
			right++;
			if (count == 0)
				result.add(left);
			if (right - left == p.length()) {
				if (hash[s.charAt(left)] >= 0) {
					count++;
				}
				hash[s.charAt(left)]++;
				left++;
			}
		}
        return result;
	}
*/

/**
 * 2017/7/10
 * 234. Palindrome Linked List
 * 
    public boolean isPalindrome(ListNode head) {
        if(head == null)return true;
        else if(head.next == null)return true;
        else if(head.next.next == null){
        	if(head.val == head.next.val)return true;
        	else return false;
        }
        ListNode temp = head;
        int sumNum = 0;
        while(temp!= null){
        	sumNum++;
        	temp = temp.next;
        }
        ListNode pre = head;
        head = head.next;
        pre.next = null;
        for(int i = 0;i < (sumNum - 1) / 2;i++){
        	temp = head.next;
        	head.next = pre;
        	pre = head;
        	head = temp;
        }
        if(sumNum % 2 == 1){
        	pre = pre.next;
        }
        while(pre != null){
        	if(pre.val != head.val)return false;
        	pre = pre.next;
        	head = head.next;
        }
        return true;
    }
*/

/**
 * 2017/7/12
 * 219. Contains Duplicate II
 * 
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer,Integer> map = new HashMap<Integer,Integer>();
        for(int i = 0;i < nums.length;i++){
        	if(map.containsKey(nums[i])){
        		if(Math.abs(i - map.get(nums[i])) <= k){
        			return true;
        		}
        		else map.put(nums[i], i);
        	}
        	else{
        		map.put(nums[i], i);
        	}
        }
        return false;
    }
    
    
 * 2017/7/12
 * 203. Remove Linked List Elements
 * 
    public ListNode removeElements(ListNode head, int val) {
        if(head == null)return head;
        while(head != null && head.val == val){
        	head = head.next;
        }
        ListNode temp = head;
        ListNode pre = head;
        while(temp != null){
        	if(temp.val == val){
        		pre.next = temp.next;
        		temp = temp.next;
        	}
        	else{
            	pre = temp;
            	temp = temp.next;
        	}
        }
        return head;
    }
    
    
 * 2017/7/12
 * 67. Add Binary
 * 
public class Solution {
    public String addBinary(String a, String b) {
        String areve = reverse(a);
        String breve = reverse(b);
        StringBuffer temp = new StringBuffer("");
        boolean flag = false;
        int i = 0;
        for(i = 0;i < Math.min(areve.length(), breve.length());i++){
        	if(areve.charAt(i) == '0' && breve.charAt(i) == '0' && !flag){
        		temp.append('0');
        		flag = false;
        	}
        	else if((areve.charAt(i) == '0' && breve.charAt(i) == '1' && !flag) || 
        			(areve.charAt(i) == '1' && breve.charAt(i) == '0' && !flag) ||
        			(areve.charAt(i) == '0' && breve.charAt(i) == '0' && flag)){
        		temp.append('1');
        		flag = false;
        	}
        	else if((areve.charAt(i) == '1' && breve.charAt(i) == '1' && !flag) || 
        			(areve.charAt(i) == '0' && breve.charAt(i) == '1' && flag) ||
        			(areve.charAt(i) == '1' && breve.charAt(i) == '0' && flag)){
        		temp.append('0');
        		flag = true;
        	}
        	else{
        		temp.append('1');
        		flag = true;
        	}
        }
        if(areve.length() == breve.length()){
        	if(flag){
        		temp.append('1');
        	}
        	return reverse(temp.toString());
        }
        else if(areve.length() > breve.length()){
        	for(;i < areve.length();i++){
        		if(areve.charAt(i) == '0' && !flag){
        			temp.append('0');
        		}
        		else if((areve.charAt(i) == '0' && flag) || (areve.charAt(i) == '1' && !flag)){
        			temp.append('1');
        			flag = false;
        		}
        		else if(areve.charAt(i) == '1' && flag){
        			temp.append('0');
        		}
        	}
        	if(flag){
        		temp.append('1');
        	}
        	return reverse(temp.toString());
        }
        else{
        	for(;i < breve.length();i++){
        		if(breve.charAt(i) == '0' && !flag){
        			temp.append('0');
        		}
        		else if((breve.charAt(i) == '0' && flag) || (breve.charAt(i) == '1' && !flag)){
        			temp.append('1');
        			flag = false;
        		}
        		else if(breve.charAt(i) == '1' && flag){
        			temp.append('0');
        		}
        	}
        	if(flag){
        		temp.append('1');
        	}
        	return reverse(temp.toString());
        }
    }
    
    String reverse(String a){
    	StringBuffer temp = new StringBuffer("");
    	for(int i = a.length() - 1;i >= 0;i--){
    		temp.append(a.charAt(i));
    	}
    	return temp.toString();
    }
}


 * 2017/7/12
 * 88. Merge Sorted Array
 * 
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int sum = n + m - 1;
        int i = m - 1;
        int j = n - 1;
		while (sum >= 0) {
			if (i >= 0 && j >= 0) {
				if (nums1[i] > nums2[j]) {
					nums1[sum--] = nums1[i--];
				} else {
					nums1[sum--] = nums2[j--];
				}
			}
			else if(i == -1){
				nums1[sum--] = nums2[j--];
			}
			else if(j == -1){
				nums1[sum--] = nums1[i--];
			}
		}
    }
    
    
 * 2017/7/12
 * 58. Length of Last Word
 * 
    public int lengthOfLastWord(String s) {
    	if(s.length() == 0)return 0;
    	int sum = 0;
    	int index = s.length() - 1;
    	while(index >= 0 && s.charAt(index) == ' '){
    		index--;
    	}
    	if(index == -1)return 0;
    	while(index >= 0 && s.charAt(index--) != ' '){
    		sum++;
    	}
    	return sum;
    }
*/

/**
 * 2017/7/13
 * 14. Longest Common Prefix
 * 
    public String longestCommonPrefix(String[] strs) {
        if(strs.length == 0)return "";
        String temp = strs[0];
        for(int i = 1;i < strs.length;i++){
        	temp = sameString(strs[i] , temp);
        }
        return temp;
    }
    
    public String sameString(String str1,String str2){
    	int i = 0;
    	for(;i < Math.min(str1.length(), str2.length());i++){
    		if(str1.charAt(i) != str2.charAt(i))
    			break;
    	}
    	return str1.substring(0,i);
    }
*/

/**
 * 2017/7/14
 * 633. Sum of Square Numbers
 * 
    public boolean judgeSquareSum(int c) {
        Set<Integer> store = new HashSet<Integer>();
        for(int i = 0;i <= Math.sqrt(c);i++){
        	store.add(i * i);
            if(store.contains(c - i * i)){
        		return true;
        	}
        }
        return false;
    }
    
    
 * 2017/7/14
 * 160. Intersection of Two Linked Lists
 * 
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int a = lengthOfListNode(headA);
        int b = lengthOfListNode(headB);
        while(a > b){
        	headA = headA.next;
        	a--;
        }
        while(b > a){
        	headB = headB.next;
        	b--;
        }
        while(headA != headB){
        	headA = headA.next;
        	headB = headB.next;
        }
        return headA;
    }
    
    public int lengthOfListNode(ListNode temp){
    	int length = 0;
    	while(temp != null){
    		temp = temp.next;
    		length++;
    	}
    	return length;
    }
    
    
 * 2017/7/14
 * 605. Can Place Flowers
 * 
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        if(n <= 0)return true;
        if(flowerbed.length == 0)return false;
        if(flowerbed.length == 1){
        	if(flowerbed[0] == 0){
        		if(n <= 1)
        			return true;
        		else
        			return false;
        	}
        	else return false;
        }
        if(flowerbed[0] == 0 && flowerbed[1] == 0){
        	flowerbed[0] = 1;
        	n--;
        }
        for(int i = 1;i < flowerbed.length - 1;i++){
        	if(flowerbed[i - 1] == 0 && flowerbed[i] == 0 && flowerbed[i + 1] == 0){
        		flowerbed[i] = 1;
        		n--;
        		i++;
        		if(n <= 0)return true;
        	}
        }
        if(flowerbed[flowerbed.length - 1] == 0 && flowerbed[flowerbed.length - 2] == 0){
        	n--;
        }
        if(n <= 0)return true;
        return false;
    }
    
    
 * 2017/7/14
 * 581. Shortest Unsorted Continuous Subarray
 * 
    public int findUnsortedSubarray(int[] nums) {
        int[] temp = nums.clone();
        Arrays.sort(temp);
        int left = nums.length - 1;
        int right = 0;
        for(int i = 0;i < temp.length;i++){
        	if(temp[i] != nums[i]){
        		left = i;
        		break;
        	}
        }
        for(int i = temp.length - 1;i >= 0;i--){
        	if(temp[i] != nums[i]){
        		right = i;
        		break;
        	}
        }
        if(right <= left)return 0;
        return (right - left + 1);
    }
*/

/**
 * 2017/7/15
 * 400. Nth Digit
 * 
	public int findNthDigit(int n) {
	    int start = 1;
	    int length = 1;
	    long count = 9;//注意此处为LONG
	    while(n > length * count){
			n -= length * count;
	    	length++;
			count *= 10;
			start *= 10;
	    }
	    start += (n - 1) / length;
		String s = Integer.toString(start);
		return Character.getNumericValue(s.charAt((n - 1) % length));
	}
*/

/**
 * 2017/7/18
 * 190. Reverse Bits
 * 
	public int reverseBits(int n) {
	    int result = 0;
	    for (int i = 0; i < 32; i++) {
	        result += n & 1;
	        n >>>= 1;   // CATCH: must do unsigned shift
	        if (i < 31) // CATCH: for last digit, don't shift!
	            result <<= 1;
	    }
	    return result;
	}
*/

/**
 * 2017/7/20
 * 28. Implement strStr()
 * 
    public int strStr(String haystack, String needle) {
    	if(needle.length() == 0)return 0;
        if(haystack.length() < needle.length())return -1;
        int m = needle.length();
        //计算特征向量
        int[] N = new int[m];
        N[0] = 0;
        for(int i = 1;i < m;i++){
        	int k = N[i - 1];
        	while(k > 0 && needle.charAt(i) != needle.charAt(k)){
        		k = N[k - 1];
        	}
        	if(needle.charAt(i) == needle.charAt(k)){
        		N[i] = k + 1;
        	}
        	else{
        		N[i] = 0;
        	}
        }
        //KMP模式匹配算法
        int i;
        int j = 0;
        for(i = 0;i < haystack.length();i++){
        	while(needle.charAt(j) != haystack.charAt(i) && j > 0){
        		j = N[j - 1];
        	}
        	if(needle.charAt(j) == haystack.charAt(i)){
        		j++;
        	}
        	if(j == needle.length()){
        		return (i - j + 1);
        	}
        }
        return -1;
    }
    
    
 * 2017/7/20
 * 414. Third Maximum Number
 * 
    public int thirdMax(int[] nums) {
        int first = Integer.MIN_VALUE;
        int second = Integer.MIN_VALUE;
        int third = Integer.MIN_VALUE;
        int temp = 0;
        for(int i = 0;i < nums.length;i++){
        	if(nums[i] == first || nums[i] == second || nums[i] == third){
        		continue;
        	}
        	if(nums[i] > third){
        		third = nums[i];
        		if(third > second){
        			temp = second;
        			second = third;
        			third = temp;
        			if(second > first){
        				temp = second;
        				second = first;
        				first = temp;
        			}
        		}
        	}
        }
        int num = 0;
        Set<Integer> set = new HashSet<Integer>();
        for(int i = 0;i < nums.length;i++){
        	if(!set.contains(nums[i])){
        		set.add(nums[i]);
        		num++;
        		if(num == 3)break;
        	}
        }
        if(num < 3)return first;
        return third;
    }
*/


/**
 * 2017/7/22
 * 475. Heaters
 * 
    public int findRadius(int[] houses, int[] heaters) {
        int result = Integer.MIN_VALUE;
        Arrays.sort(heaters);
        for(int house:houses){
        	int index = Arrays.binarySearch(heaters, house);
        	if(index < 0){
        		index = -1 - index;
        	}
        	int temp1 = index - 1 >= 0?  house - heaters[index - 1]:Integer.MAX_VALUE;
        	int temp2 = index < heaters.length? heaters[index] - house:Integer.MAX_VALUE;
        	result = Math.max(Math.min(temp1, temp2), result);
        }
        return result;
    }
*/


/**
 * 2017/7/23
 * 532. K-diff Pairs in an Array
 * 
    public int findPairs(int[] nums, int k) {
        int result = 0;
        if(k < 0)return result;
        if(k == 0){
        	Map<Integer,Integer> map = new HashMap<Integer,Integer>();
        	for(int i = 0;i < nums.length;i++){
        		if(map.containsKey(nums[i])){
        			map.put(nums[i], map.get(nums[i]) + 1);
        		}
        		else{
        			map.put(nums[i], 1);
        		}
        	}
        	Set<Integer> keys = map.keySet();
        	for(int key : keys){
        		if(map.get(key) >= 2){
        			result++;
        		}
        	}
        }
    	else{
    		Set<Integer> set = new HashSet<Integer>();
    		for(int i = 0;i < nums.length;i++){
    			set.add(nums[i]);
    		}
    		for(int key:set){
    			if(set.contains(key - k)){
    				result++;
    			}
    			if(set.contains(key + k)){
    				result++;
    			}
    		}
    		result = result / 2;
    	}
    	return result;
    }
    
    
 * 2017/7/23
 * 204. Count Primes
 * 
    public int countPrimes(int n) {
        if(n < 2)return 0;
        else{
            ArrayList<Integer> array = new ArrayList<Integer>();
            int flag = 0;
            for(int i = 2;i < n;i++){
            	flag = 1;
            	for(int j = 0;j < array.size();j++){
            		if(i % array.get(j) == 0){
            			flag = 0;
            			break;
            		}
            		if(array.get(j) * array.get(j) > i){
            			break;
            		}
            	}
            	if(flag == 1){
            		array.add(i);
            	}
            }
            return array.size();
        }
    }
    
    
 * 2017/7/23
 * 69. Sqrt(x)   
 * 
    public int mySqrt(int x) {
        return (int)(Math.sqrt(x) + 0.1);
    }
    
    
 * 2017/7/23
 * 125. Valid Palindrome
 * 
    public boolean isPalindrome(String s) {
        String temp = s.toLowerCase();
        StringBuffer str = new StringBuffer();
        for(int i = 0;i < temp.length();i++){
        	if((temp.charAt(i) <= 'z' && temp.charAt(i) >= 'a') || (temp.charAt(i) <= '9' && temp.charAt(i) >= '0')){
        		str.append(temp.charAt(i));
        	}
        }
        return isStringPalindrome(str.toString());
    }
    
    public boolean isStringPalindrome(String s){
    	for(int i = 0;i < s.length() / 2;i++){
    		if(s.charAt(i) != s.charAt(s.length() - i - 1)){
    			return false;
    		}
    	}
    	return true;
    }
    
    
 * 2017/7/23
 * 168. Excel Sheet Column Title
 * 
    public String convertToTitle(int n) {
        StringBuffer str = new StringBuffer();
        while(n > 0){
        	if(n % 26 == 0){
        		str.append('Z');
        		n = n - 26;
        	}
        	else{
            	str.append((char)(n % 26 - 1 + 'A'));
        	}
        	n = n / 26;
        }
        return str.reverse().toString();
    }
    
    
 * 2017/7/23
 * 278. First Bad Version
 * 
    The isBadVersion API is defined in the parent class VersionControl.
    boolean isBadVersion(int version);
    public int firstBadVersion(int n) {
        if(n < 1)return 0;
        int left = 1;
        int right = n;
        int mid;
        while(left < right){
            mid = left + (right - left) / 2;
            if(isBadVersion(mid)){
                right = mid - 1;
                if(!isBadVersion(mid - 1)){
                    return mid;
                }
            }
            else{
                left = mid + 1;
            }
        }
        return left;
    }
*/


/**
 * 2017/7/24
 * 189. Rotate Array
 * 
    public void rotate(int[] nums, int k) {
        Queue<Integer> queue = new LinkedList<Integer>();
        for(int i = nums.length - k % nums.length;i < nums.length;i++){
        	queue.add(nums[i]);
        }
        for(int i = 0;i < nums.length - k % nums.length;i++){
        	queue.add(nums[i]);
        }
        int i = 0;
        while(!queue.isEmpty()){
        	nums[i++] = queue.poll();
        }
    }
    
    
 * 2017/7/24
 * 537. Complex Number Multiplication
 * 
    public String complexNumberMultiply(String a, String b) {
        int indexA = a.indexOf('+');
        int numA1 = Integer.parseInt(a.substring(0, indexA));
        int indexB = a.indexOf('i');
        int numA2 = Integer.parseInt(a.substring(indexA + 1,indexB));
        indexA = b.indexOf('+');
        int numB1 = Integer.parseInt(b.substring(0, indexA));
        indexB = b.indexOf('i');
        int numB2 = Integer.parseInt(b.substring(indexA + 1,indexB));
        int numC1 = numA1 * numB1 - numA2 * numB2;
        int numC2 = numA1 * numB2 + numA2 * numB1;
        return numC1 + "+" + numC2 + 'i';
    }
*/


/**
 * 2017/7/25
 * 7. Reverse Integer
 * 
    public int reverse(int x) {
        int flag = 1;
        if(x < 0)flag = -1;//记录是否为负数
        x = Math.abs(x);
        int result = 0;
        int pre = 0;
        while(x > 0){
        	pre = result;
        	result = result * 10 + x % 10;
        	x = x / 10;
        }
        if(result / 10 != pre){
        	return 0;
        }
        if(flag == 1)return result;
        else
        	return -result;
    }
    
    
 * 2017/7/25
 * 479. Largest Palindrome Product
 * 
    public int largestPalindrome(int n) {
    	if(n == 1)return 9;
        int max = (int)Math.pow(10, n) - 1;
        long result;
        for(int i = max;i > max / 10;i--){
        	result = (long)Long.parseLong(i + new StringBuffer().append(i).reverse().toString());
        	for(long j = max;j * j >= result;j--){
        		if(result % j == 0){
        			return (int)(result % 1337);
        		}
        	}
        }
        return 0;
    }
    
    
 * 2017/7/25
 * 419. Battleships in a Board
 * 
    public int countBattleships(char[][] board) {
        int count = 0;
        if(board.length == 0 || board[0].length == 0)return count;
        for(int i = 0;i < board.length;i++){
        	for(int j = 0;j < board[0].length;j++){
        		if(board[i][j] == 'X' && (i == 0 || board[i - 1][j] == '.') && (j == 0 || board[i][j - 1] == '.'))
        			count++;
        	}
        }
        return count;
    }
    
    
 * 2017/7/25
 * 338. Counting Bits
 * 
    public int[] countBits(int num) {
        int[] result = new int[num + 1];
        for(int i = 0;i <= num;i++){
        	result[i] = countBit(i);
        }
        return result;
    }
    public int countBit(int num){
    	int count = 0;
    	while(num > 0){
    		count += num & 1;
    		num = num >> 1;
    	}
    	return count;
    }
    
    
 * 2017/7/25
 * 647. Palindromic Substrings
 * 
    public int countSubstrings(String s) {
        int count = 0;
    	for(int i = 0;i < s.length();i++){
    		for(int j = i;j < s.length();j++){
    			if(isPalindromic(s.substring(i,j + 1)))
    				count++;
    		}
    	}
        return count;
    }
    public boolean isPalindromic(String s){
    	for(int i = 0;i < s.length() / 2;i++){
    		if(s.charAt(i) != s.charAt(s.length() - 1 - i))
    			return false;
    	}
    	return true;
    }
    
    
 * 2017/7/25
 * 513. Find Bottom Left Tree Value
 * 
    public int findBottomLeftValue(TreeNode root) {
    	if(root == null)return 0;
        Queue<TreeNode> que = new LinkedList<TreeNode>();
        que.offer(root);
        int result = root.val;
        while(!que.isEmpty()){
        	result = que.peek().val;
        	int size = que.size();
        	for(int i = 0;i < size;i++){
        		root = que.poll();
        		if(root.left != null)que.add(root.left);
        		if(root.right != null)que.add(root.right);
        	}
        }
        return result;
    }
*/

package exercise;

public class Solution_July8th_To_July25th {

}