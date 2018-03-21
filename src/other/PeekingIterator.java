package other;

import java.util.Iterator;

public class PeekingIterator implements Iterator<Integer> {
	Iterator<Integer> iter;
	Integer next;
	
	public PeekingIterator(Iterator<Integer> iterator) {
	    // initialize any member here.
	    iter = iterator;
	    if(iter.hasNext()){
	    	next = iter.next();
	    }
	}

    // Returns the next element in the iteration without advancing the iterator.
	public Integer peek() {
		return next;
	}
	
	@Override
	public boolean hasNext() {
		// TODO 自动生成的方法存根
		return next != null;
	}

	@Override
	public Integer next() {
		// TODO 自动生成的方法存根
		Integer res = next;
		next = iter.hasNext() ? iter.next() : null;
		return res;
	}
}
