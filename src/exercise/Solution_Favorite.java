/**
 * @author jhy
 * code for my favorite question 
 * 0 questions
 */


/**
 * 2017/11/14
 * 53. Maximum Subarray
 * 
	public int maxSubArray(int[] nums) {
		int[] dp = new int[nums.length];
		int max = nums[0];
		dp[0] = nums[0];
		for (int i = 1; i < nums.length; i++) {
			dp[i] = nums[i] + (dp[i - 1] > 0 ? dp[i - 1] : 0);
			max = Math.max(max, dp[i]);
		}
		return max;
	}
	
 * 2017/11/14
 * 121. Best Time to Buy and Sell Stock
 * 
    public int maxProfit(int[] prices) {
        if(prices.length == 0)return 0;
        int max = 0;
        int min = prices[0];
        for(int i = 1;i < prices.length;i++){
        	max = Math.max(max, prices[i] - min);
        	min = Math.min(min, prices[i]);
        }
        return max;
    }
    
 * 2017/11/14
 * 309. Best Time to Buy and Sell Stock with Cooldown
 * 
	public int maxProfit(int[] prices) {
		int sell = 0, buy = Integer.MIN_VALUE, pre_buy, pre_sell = 0;
		for (int i = 0; i < prices.length; i++) {
			pre_buy = buy;
			buy = Math.max(pre_sell - prices[i], pre_buy);
			pre_sell = sell;
			sell = Math.max(pre_buy + prices[i], pre_sell);
		}
    	return sell;
    }
 */

/**
 * 2017/11/21
 * 136. Single Number
 * 
    public int singleNumber(int[] nums) {
        for (int i = 1; i < nums.length; i++) {
			nums[0] ^= nums[i];
		}
        return nums[0];
    }
 */

/**
 * 2017/11/22
 * 260. Single Number III
 * 
    public int[] singleNumber(int[] nums) {
        int index = 0;
        for(int num:nums){
        	index ^= num;
        }
        index &= -index;
        int[] result = new int[2];
        for(int num:nums){
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
 * 2017/11/27
 * 401. Binary Watch
 * 
class Solution {
    public List<String> readBinaryWatch(int num) {
        Set<Integer> Hour = new HashSet<Integer>();
        int i = 1;
        while(i <= 8){
        	Hour.add(i);
        	i *= 2;
        }
        Set<Integer> Minute = new HashSet<Integer>();
        i = 1;
        while(i <= 32){
        	Minute.add(i);
        	i *= 2;
        }
        List<String> result = new ArrayList<String>();
        Set<Integer> hourResult = new HashSet<Integer>();
        Set<Integer> minuteResult = new HashSet<Integer>();
        for(int j = 0;j <= num;j++){
        	hourResult = binaryTime(j, Hour, 12);
        	minuteResult = binaryTime(num - j, Minute, 60);
        	for(int h:hourResult){
        		for(int m:minuteResult){
        			if(m < 10){
        				String str = new String(h + ":0" + m);
            			result.add(str);
        			}
        			else{
        				String str = new String(h + ":" + m);
            			result.add(str);
        			}
        		}
        	}
        }
        return result;
    }
    
    Set<Integer> binaryTime(int n, Set<Integer> number,int max){//maxÎªãÐÖµ,hour = 12,minute = 60
    	if(n == 0){
    		Set<Integer> result = new HashSet<Integer>();
    		result.add(0);
    		return result;
    	}
    	if(n == 1)return number;
    	Set<Integer> result = new HashSet<Integer>();
    	Set<Integer> temp = new HashSet<Integer>(number);
    	for(int i : number){
    		temp.remove(i);
    		Set<Integer> tempResult = binaryTime(n - 1, temp, max);
    		for(int j:tempResult){
    			if(i + j < max){
    				result.add(i + j);
    			}
    		}
    		temp.add(i);
    	}
    	return result;
    }
}

 */


package exercise;

public class Solution_Favorite {

}
