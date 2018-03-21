package other;

import java.util.ArrayList;
import java.util.List;
//731. My Calendar II
class MyCalendarTwo {
	List<int[]> calendar;
	List<int[]> overlap;
	
    public MyCalendarTwo() {
        overlap = new ArrayList<>();
        calendar = new ArrayList<>();
    }
    
    public boolean book(int start, int end) {
        for(int[] temp: overlap){
        	if(temp[0] < end && temp[1] > start)return false;
        }
        for(int[] temp : calendar){
        	if(temp[0] < end && temp[1] > start){
        		overlap.add(new int[]{Math.max(temp[0], start), Math.min(temp[1], end)});
        	}
        }
        calendar.add(new int[]{start,end});
        return true;
    }
}

/**
 * Your MyCalendarTwo object will be instantiated and called as such:
 * MyCalendarTwo obj = new MyCalendarTwo();
 * boolean param_1 = obj.book(start,end);
 */