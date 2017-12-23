package program;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Solution {
    public int deleteAndEarn(int[] nums) {
    	//对nums中的数据整理成数组list[2],list[0]代表数字list[1]代表该数字总和
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0;i < nums.length;i++){
        	if(map.containsKey(nums[i])){
        		map.put(nums[i], map.get(nums[i]) + nums[i]);
        	}
        	else{
        		map.put(nums[i], nums[i]);
        	}
        }
        List<int[]> list = new ArrayList<>();
        for(int key : map.keySet()){
        	int[] temp = new int[2];
        	temp[0] = key;
        	temp[1] = map.get(key);
        	list.add(temp);
        }
        //对数组按升序排列
        Collections.sort(list,new Comparator<int[]>() {
            public int compare(int[] o1, int[] o2) {  
                return o1[0] - o2[0];
            }  
		});
        if(list.size() == 0)return 0;
        if(list.size() == 1)return list.get(0)[1];
        int[] dp = new int [list.size()];
        //动态规划解决问题
        dp[0] = list.get(0)[1];
        if(list.get(1)[0] == list.get(0)[0] + 1){
        	dp[1] = Math.max(list.get(1)[1], list.get(0)[1]);
        }
        else{
        	dp[1] = list.get(1)[1] + list.get(0)[1];
        }
        for(int i = 2;i < list.size();i++){
        	if(list.get(i)[0] == list.get(i - 1)[0] + 1){
        		dp[i] = Math.max(dp[i - 1], dp[i - 2] + list.get(i)[1]);
        	}
        	else{
        		dp[i] = dp[i - 1] + list.get(i)[1];
        	}
        }
        return dp[list.size() - 1];
    }
}
