package other;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
//460. LFU Cache
class LFUCache {
	Map<Integer, Integer> values;
	Map<Integer, Integer> counts;
	Map<Integer, LinkedHashSet<Integer>> lists;
	int cap;
	int min = -1;
	
    public LFUCache(int capacity) {
        cap = capacity;
        values = new HashMap<>();
        counts = new HashMap<>();
        lists = new HashMap<>();
        lists.put(1, new LinkedHashSet<>());
    }
    
    public int get(int key) {
        if(!values.containsKey(key))
        	return -1;
        int count = counts.get(key);
        counts.put(key, count + 1);
        lists.get(count).remove(key);
        if(count == min && lists.get(count).size() == 0)
        	min++;
        if(!lists.containsKey(count + 1))
        	lists.put(count + 1, new LinkedHashSet<>());
        lists.get(count + 1).add(key);
        return values.get(key);
    }
    
    public void put(int key, int value) {
    	if(cap <= 0)
    		return;
        if(values.containsKey(key)){
        	values.put(key, value);
        	get(key);
        	return;
        }
        if(values.size() >= cap){
        	int evit = lists.get(min).iterator().next();
        	values.remove(evit);
        	lists.get(min).remove(evit);
        }
        values.put(key, value);
        counts.put(key, 1);
        min = 1;
        lists.get(1).add(key);
    }
}

/**
 * Your LFUCache object will be instantiated and called as such:
 * LFUCache obj = new LFUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
