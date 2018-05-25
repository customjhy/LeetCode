package other;

public class SegmentTree {
	//�ݹ齨���߶�������ɲ���
	public final int maxn = 10007;
	public int[] A = new int[maxn];
	public int n;
	public int[] Sum = new int[maxn << 2];
	public int[] Add = new int[maxn << 2];
	
	//Pushup�������½����Ϣ������Ϊ���
	public void pushUp(int rt){
		Sum[rt] = Sum[rt << 1] + Sum[rt << 1 | 1];
	}
	
	//build����
	public void build(int l, int r, int rt){
		if(l == r){
			Sum[rt] = A[l];
			return;
		}
		int m = (l + r) >> 1;
		//���ҵݹ�
		build(l, m, rt << 1);
		build(m + 1, r, rt << 1 | 1);
		pushUp(rt);
	}
	
	//���޸ģ�l,rΪ��ǰ��㣬rtΪ�ڵ���
	public void updateNode(int L, int R, int C, int l, int r, int rt){
		if(l == r){//��Ҷ�ڵ�
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
	
	//�����޸�
	public void updateInterval(int L, int R, int C, int l, int r, int rt){
		if(L <= l && r <= R){
			Sum[rt] += C * (r - l + 1);
			Add[rt] += C;//����add��ǣ���ʾ�������Sum��ȷ����������Ҫ����
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
	
	//���Ʊ��
	public void pushDown(int rt, int ln, int rn){
		//ln��rnΪ������������������
		if(Add[rt] != 0){
			Add[rt << 1] += Add[rt];
			Add[rt << 1 | 1] += Add[rt];
			Sum[rt << 1] += Add[rt] * ln;
			Sum[rt << 1 | 1] += Add[rt] * rn;
			Add[rt] = 0;
		}
	}
	
	//�����ѯ����
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
