package other;

import java.util.HashMap;
import java.util.Map;

class MapSum {
	Map<String, Integer> map = new HashMap<>();
	
    /** Initialize your data structure here. */
    public MapSum() {
        
    }
    
    public void insert(String key, int val) {
        map.put(key, val);
    }
    
    public int sum(String prefix) {
        int res = 0;
        for(String str : map.keySet()){
        	if(isSubstring(str, prefix)){
        		res += map.get(str);
        	}
        }
        return res;
    }
    
    public boolean isSubstring(String str,String sub){
    	if(str.length() < sub.length())return false;
    	int i = 0;
    	while(i < sub.length()){
    		if(str.charAt(i) != sub.charAt(i)){
    			return false;
    		}
    		i++;
    	}
    	return true;
    }
}

/**
 * Your MapSum object will be instantiated and called as such:
 * MapSum obj = new MapSum();
 * obj.insert(key,val);
 * int param_2 = obj.sum(prefix);
 */
