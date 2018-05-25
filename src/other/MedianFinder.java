package other;

import java.util.Collections;
import java.util.PriorityQueue;
//295. Find Median from Data Stream
class MedianFinder {
	PriorityQueue<Long> big = new PriorityQueue<Long>();
	PriorityQueue<Long> small = new PriorityQueue<Long>(Collections.reverseOrder());

    /** initialize your data structure here. */
    public MedianFinder() {
    	
    }
    
    public void addNum(int num) {
        big.add((long)num);
        small.add(big.poll());
        if(small.size() > big.size()){
            big.add(small.poll());
        }
    }
    
    public double findMedian() {
        return big.size() == small.size()?
            ((double)big.peek() + (double)small.peek()) / 2:
            big.peek();
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */