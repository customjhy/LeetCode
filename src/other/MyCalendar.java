package other;

import java.util.ArrayList;
import java.util.List;
//729. My Calendar I
class MyCalendar {
	List<Integer[]> list;
	
    public MyCalendar() {
        list = new ArrayList<>();
    }
    
    public boolean book(int start, int end) {
        for(Integer[] cal : list){
        	if((cal[0].intValue() <= start && start < cal[1].intValue()) ||
        			(cal[0].intValue() < end && end <= cal[1].intValue())||
        			(cal[0].intValue() > start && end >= cal[1].intValue())){
        		return false;
        	}
        }
        Integer[] add = new Integer[2];
        add[0] = start;
        add[1] = end;
    	list.add(add); 
    	return true;
    }
}

/**
 * Your MyCalendar object will be instantiated and called as such:
 * MyCalendar obj = new MyCalendar();
 * boolean param_1 = obj.book(start,end);
 */
