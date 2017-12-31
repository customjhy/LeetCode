package other;

class MagicDictionary {
	class TrieNode{
		TrieNode[] childTrie = new TrieNode[26];
		boolean isWord;
		public TrieNode() {
			// TODO 自动生成的构造函数存根
		}
	}
	
	TrieNode root;
	
	/** Initialize your data structure here. */
    public MagicDictionary() {
        root = new TrieNode();
    }
    
    /** Build a dictionary through a list of words */
    public void buildDict(String[] dict) {
        TrieNode temp = root;
    	for(String str : dict){
        	temp = root;
        	for(int i = 0;i < str.length();i++){
        		if(temp.childTrie[str.charAt(i) - 'a'] == null){
        			temp.childTrie[str.charAt(i) - 'a'] = new TrieNode();
        		}
        		temp = temp.childTrie[str.charAt(i) - 'a'];
        	}
        	temp.isWord = true;
        }
    }
    
    public boolean isEqual(String str){
    	TrieNode temp = root;
    	for(int i = 0;i < str.length();i++){
    		temp = temp.childTrie[str.charAt(i) - 'a'];
    		if(temp == null)return false;
    	}
    	return temp.isWord;
    }
    
    /** Returns if there is any word in the trie that equals to the given word after modifying exactly one character */
    public boolean search(String word) {
        char[] str = word.toCharArray();
        for(int i = 0;i < str.length;i++){
        	for(char j = 'a';j <= 'z';j++){
        		if(j == str[i])continue;
        		char letter = str[i];
        		str[i] = j;
        		if(isEqual(new String(str)))return true;
        		str[i] = letter;
        	}
        }
        return false;
    }
}

/**
 * Your MagicDictionary object will be instantiated and called as such:
 * MagicDictionary obj = new MagicDictionary();
 * obj.buildDict(dict);
 * boolean param_2 = obj.search(word);
 */