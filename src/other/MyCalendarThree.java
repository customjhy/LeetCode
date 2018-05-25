package other;

import java.util.TreeMap;

//732. My Calendar III
class MyCalendarThree {
	TreeMap<Integer, Integer> timeline = new TreeMap<>();
	
    public MyCalendarThree() {
        
    }
    
    public int book(int start, int end) {
        timeline.put(start, timeline.getOrDefault(start, 0) + 1);
        timeline.put(end, timeline.getOrDefault(end, 0) - 1);
        int max = 0;
        int sum = 0;
        for(int time : timeline.values()){
        	sum += time;
        	max = Math.max(max, sum);
        }
        return max;
    }
}


/**
 * Your MyCalendarThree object will be instantiated and called as such:
 * MyCalendarThree obj = new MyCalendarThree();
 * int param_1 = obj.book(start,end);
 */