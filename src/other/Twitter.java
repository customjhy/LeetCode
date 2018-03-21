package other;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
//355. Design Twitter
class Twitter {
	int time;
	Map<Integer, List<int[]>> post;
	Map<Integer, List<Integer>> follow;
	
    /** Initialize your data structure here. */
    public Twitter() {
        time = 0;
        post = new HashMap<>();
        follow = new HashMap<>();
    }
    
    /** Compose a new tweet. */
    public void postTweet(int userId, int tweetId) {
        if(post.containsKey(userId)){
        	post.get(userId).add(new int[]{tweetId,time++});
        }
        else{
        	List<int[]> list = new ArrayList<>();
        	list.add(new int[]{tweetId,time++});
        	post.put(userId, list);
        }
    }
    
    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    public List<Integer> getNewsFeed(int userId) {
        List<int[]> com = new ArrayList<>();
        List<int[]> temp;
        Set<Integer> use = new HashSet<>();
		if (post.containsKey(userId)) {
			temp = post.get(userId);
			if (temp != null) {
				for (int i = temp.size() - 1; i >= 0 && i >= temp.size() - 10; i--) {
					com.add(temp.get(i));
				}
			}
		}
		use.add(userId);
		if (follow.containsKey(userId)) {
			for (int followId : follow.get(userId)) {
				if (!use.contains(followId)) {
					temp = post.get(followId);
					if (temp != null) {
						for (int i = temp.size() - 1; i >= 0 && i >= temp.size() - 10; i--) {
							com.add(temp.get(i));
						}
					}
				}
				use.add(followId);
			}
		}
        Collections.sort(com, new Comparator<int[]>() {
			public int compare(int[] o1, int[] o2) {
				return o2[1] - o1[1];
			}
		});
        List<Integer> res = new ArrayList<>();
        for(int i = 0;i < 10 && i < com.size();i++){
        	res.add(com.get(i)[0]);
        }
        return res;
    }
    
    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    public void follow(int followerId, int followeeId) {
        if(follow.containsKey(followerId)){
        	follow.get(followerId).add(followeeId);
        }
        else{
        	List<Integer> list = new ArrayList<>();
        	list.add(followeeId);
        	follow.put(followerId, list);
        }
    }
    
    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    public void unfollow(int followerId, int followeeId) {
        if(follow.containsKey(followerId)){
        	List<Integer> list = follow.get(followerId);
        	int index = list.lastIndexOf(followeeId);
        	if(index != -1)list.remove(index);
        }
    }
}

/**
 * Your Twitter object will be instantiated and called as such:
 * Twitter obj = new Twitter();
 * obj.postTweet(userId,tweetId);
 * List<Integer> param_2 = obj.getNewsFeed(userId);
 * obj.follow(followerId,followeeId);
 * obj.unfollow(followerId,followeeId);
 */