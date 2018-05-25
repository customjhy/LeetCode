package other;

import java.util.Deque;
//146. LRU Cache
class LRUCache {
	class Node{
		int key;
		int value;
		Node pre;
		Node next;
		public Node(int k, int v) {
			key = k;
			value = v;
		}
	}
	
	Node head;
	Node tail;
	int len;
	int cur;
	
	public Node search(int key){
		Node temp = head.next;
		while(temp != tail){
			if(temp.key == key)return temp;
			temp = temp.next;
		}
		return null;
	}
	
	public Node remove(Node node){
		node.pre.next = node.next;
		node.next.pre = node.pre;
		return node;
	}
	
	public boolean insect(Node node){
		node.next = head.next;
		head.next.pre = node;
		node.pre = head;
		head.next = node;
		return true;
	}
	
    public LRUCache(int capacity) {
        len = capacity;
        cur = 0;
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head.next = tail;
        tail.pre = head;
        tail.next = null;
        head.pre = null;
    }
    
    public int get(int key) {
        Node temp = search(key);
        if(temp == null)return -1;
        remove(temp);
        insect(temp);
        return temp.value;
    }
    
    public void put(int key, int value) {
        Node temp = search(key);
        if(temp != null){
        	temp.value = value;
        	remove(temp);
        	insect(temp);
        	return;
        }
		while (cur >= len) {
			cur--;
			tail.pre.pre.next = tail;
			tail.pre = tail.pre.pre;
		}
		insect(new Node(key, value));
		cur++;
	}
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */