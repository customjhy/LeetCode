package other;
//211. Add and Search Word - Data structure design
public class WordDictionary {
	class TrieNode{
		boolean isWord;
		TrieNode[] trie = new TrieNode[26];
	}
	TrieNode root;
	
    /** Initialize your data structure here. */
    public WordDictionary() {
        root = new TrieNode();
    }
    
    /** Adds a word into the data structure. */
    public void addWord(String word) {
        if(word == null || word.length() == 0)return;
        TrieNode temp = root;
        for(char ch : word.toCharArray()){
        	if(root.trie[ch - 'a'] == null){
        		root.trie[ch - 'a'] = new TrieNode();
        	}
        	root = root.trie[ch - 'a'];
        }
        root.isWord = true;
        root = temp;
    }
    
    /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
    public boolean search(String word) {
        return helpSearch(root, word);
    }
    
    public boolean helpSearch(TrieNode root, String word) {
    	if(word == null || word.length() == 0){
    		if(root == null)return false;
    		return root.isWord;
    	}
    	TrieNode temp = root;
    	for(int  i = 0;i < word.length();i++){
    		char ch = word.charAt(i);
    		if(ch != '.'){
    			if(temp.trie[ch - 'a'] == null){
    				return false;
    			}
    			else temp = temp.trie[ch - 'a'];
    		}
    		else{
    			String next = word.substring(i + 1);
    			for(int j = 0;j < 26;j++){
    				if(temp.trie[j] != null && helpSearch(temp.trie[j], next))return true;
    			}
    			return false;
    		}
    	}
    	return temp.isWord;
	}
}

/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary obj = new WordDictionary();
 * obj.addWord(word);
 * boolean param_2 = obj.search(word);
 */