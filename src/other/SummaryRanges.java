package other;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;
//352. Data Stream as Disjoint Intervals
class SummaryRanges {
	public class Interval {
		int start;
		int end;

		Interval() {
			start = 0;
			end = 0;
		}

		Interval(int s, int e) {
			start = s;
			end = e;
		}
	}
	
	TreeMap<Integer, Integer> map;
	
	/** Initialize your data structure here. */
    public SummaryRanges() {
    	map = new TreeMap<>();
    }
    
    public void addNum(int val) {
    	if(map.containsKey(val))return;
        Integer cell = map.ceilingKey(val);
        Integer floor = map.floorKey(val);
        if(floor != null && map.get(floor) >= val)return;
        if(cell != null && floor != null && val == map.get(floor) + 1 && val + 1 == cell){
        	map.put(floor, map.get(cell));
        	map.remove(cell);
        }
        else if(floor != null && val == map.get(floor) + 1){
        	map.put(floor, val);
        }
        else if(cell != null && val + 1 == cell){
        	map.put(val, map.get(cell));
        	map.remove(cell);
        }
        else{
        	map.put(val, val);
        }
    }
    
    public List<Interval> getIntervals() {
        List<Interval> res = new ArrayList<>();
        for(int i : map.keySet()){
        	res.add(new Interval(i, map.get(i)));
        }
        return res;
    }
}

/**
 * Your SummaryRanges object will be instantiated and called as such:
 * SummaryRanges obj = new SummaryRanges();
 * obj.addNum(val);
 * List<Interval> param_2 = obj.getIntervals();
 */
