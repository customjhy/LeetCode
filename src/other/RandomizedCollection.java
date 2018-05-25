package other;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
//381. Insert Delete GetRandom O(1) - Duplicates allowed
public class RandomizedCollection {
	public List<Integer> array = new ArrayList<>();
	public Map<Integer, LinkedHashSet<Integer>> map = new HashMap<>();
	
    /** Initialize your data structure here. */
    public RandomizedCollection() {
        
    }
    
    /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
    public boolean insert(int val) {
        if(map.containsKey(val)){
        	map.get(val).add(array.size());
        	array.add(val);
        	return false;
        }else{
        	LinkedHashSet<Integer> temp = new LinkedHashSet<>();
        	temp.add(array.size());
        	map.put(val, temp);
        	array.add(val);
        	return true;
        }
    }
    
    /** Removes a value from the collection. Returns true if the collection contained the specified element. */
    public boolean remove(int val) {
        if(!map.containsKey(val)){
        	return false;
        }
        int removeValIndex = map.get(val).iterator().next();
        //array最后一个数字移动到removeValIndex处
        int lastNum = array.get(array.size() - 1);
        LinkedHashSet<Integer> lastNumSet = map.get(lastNum);
        array.set(removeValIndex, lastNum);
        map.get(val).remove((Integer)removeValIndex);
        if(array.size() - 1 != removeValIndex){
        	lastNumSet.remove((Integer)(array.size() - 1));
        	lastNumSet.add(removeValIndex);
        }
        //map.get(val).remove((Integer)removeValIndex);
        //该语句不可以放在这个位置，map.get(val)与lastNumSet可能重复。
        array.remove(array.size() - 1);
        //map中对val的linkedHashSet进行更新
        if(map.get(val).isEmpty())map.remove(val);
        return true;
    }
    
    /** Get a random element from the collection. */
    public int getRandom() {
    	return array.get((int)(Math.random() * array.size()));
    }
}

/**
 * Your RandomizedCollection object will be instantiated and called as such:
 * RandomizedCollection obj = new RandomizedCollection();
 * boolean param_1 = obj.insert(val);
 * boolean param_2 = obj.remove(val);
 * int param_3 = obj.getRandom();
 */
