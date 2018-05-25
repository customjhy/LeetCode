package other;

public class SegmentTree {
	//递归建立线段树及完成操作
	public final int maxn = 10007;
	public int[] A = new int[maxn];
	public int n;
	public int[] Sum = new int[maxn << 2];
	public int[] Add = new int[maxn << 2];
	
	//Pushup函数更新结点信息，这里为求和
	public void pushUp(int rt){
		Sum[rt] = Sum[rt << 1] + Sum[rt << 1 | 1];
	}
	
	//build建树
	public void build(int l, int r, int rt){
		if(l == r){
			Sum[rt] = A[l];
			return;
		}
		int m = (l + r) >> 1;
		//左右递归
		build(l, m, rt << 1);
		build(m + 1, r, rt << 1 | 1);
		pushUp(rt);
	}
	
	//点修改，l,r为当前结点，rt为节点编号
	public void updateNode(int L, int R, int C, int l, int r, int rt){
		if(l == r){//到叶节点
			Sum[rt] += C;
			return;
		}
		int m = (l + r) >> 1;
		if(m >= L){
			updateNode(L, R, C, l, m, rt >> 1);
		}
		else{
			updateNode(L, R, C, m + 1, r, rt >> 1 | 1);
		}
		pushUp(rt);
	}
	
	//区间修改
	public void updateInterval(int L, int R, int C, int l, int r, int rt){
		if(L <= l && r <= R){
			Sum[rt] += C * (r - l + 1);
			Add[rt] += C;//增加add标记，表示本区间的Sum正确，子区间需要调整
			return;
		}
		int m = (l + r) >> 1;
		pushDown(rt, m - l + 1, r - m);
		if(L <= m){
			updateInterval(L, R, C, l, m, rt << 1);
		}
		if(m < R){
			updateInterval(L, R, C, m + 1, r, rt << 1 | 1);
		}
		pushUp(rt);
	}
	
	//下推标记
	public void pushDown(int rt, int ln, int rn){
		//ln，rn为左右子树的数字数量
		if(Add[rt] != 0){
			Add[rt << 1] += Add[rt];
			Add[rt << 1 | 1] += Add[rt];
			Sum[rt << 1] += Add[rt] * ln;
			Sum[rt << 1 | 1] += Add[rt] * rn;
			Add[rt] = 0;
		}
	}
	
	//区间查询函数
	public int query(int L, int R, int l, int r, int rt){
		if(L <= l && r <= R){
			return Sum[rt];
		}
		int m = (l + r) >> 1;
		pushDown(rt, m - l + 1, r - m);
		
		int res = 0;
		if(L <= m)res += query(L, R, l, m, rt << 1);
		if(m < R)res += query(L, R, m + 1, r, rt << 1 | 1);
		return res;
	}
}
