package other;
//304. Range Sum Query 2D - Immutable
class NumMatrix {
    private int[][] dp;
	
    public NumMatrix(int[][] matrix) {
        if(matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)return;
        int n = matrix.length + 1;
        int m = matrix[0].length + 1;
        dp = new int[n][m];
        for(int i = 1;i < n;i++){
        	for(int j = 1;j < m;j++){
        		dp[i][j] = dp[i - 1][j] + dp[i][j - 1] - dp[i - 1][j - 1] + matrix[i - 1][j - 1];
        	}
        }
    }
    
    public int sumRegion(int row1, int col1, int row2, int col2) {
        int iMin = Math.min(row1, row2);
        int iMax = Math.max(row1, row2);
        int jMin = Math.min(col1, col2);
        int jMax = Math.max(col1, col2);
        return dp[iMax + 1][jMax + 1] - dp[iMin][jMax + 1] - dp[iMax + 1][jMin] + dp[iMin][jMin];
    }
}

/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix obj = new NumMatrix(matrix);
 * int param_1 = obj.sumRegion(row1,col1,row2,col2);
 */