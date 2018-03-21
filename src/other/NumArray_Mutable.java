package other;
//307. Range Sum Query - Mutable
class NumArray_Mutable {
	//binary indexed tree--New way
	int[] nums;
	int[] BIT;
	int N;
	
	public NumArray_Mutable(int[] nums) {
		this.nums = nums;

		N = nums.length;
		BIT = new int[N + 1];
		for (int i = 0; i < N; i++)
			init(i, nums[i]);
	}

	public void init(int i, int val) {
		i++;
		while (i <= N) {
			BIT[i] += val;
			i += (i & -i);
		}
	}

	void update(int i, int val) {
		int diff = val - nums[i];
		nums[i] = val;
		init(i, diff);
	}
    
    public int sumRange(int i, int j) {
        return sum(j + 1) - sum(i);
    }
    
    public int sum(int k){
        int sum = 0;
        while (k > 0){
            sum += BIT[k];
            k -= (k & -k);
        }
        return sum;
    }
}

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray obj = new NumArray(nums);
 * obj.update(i,val);
 * int param_2 = obj.sumRange(i,j);
 */