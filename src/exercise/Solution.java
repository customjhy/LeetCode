/**
 * @author jhy
 * code from 12.31 to
 * 0 questions
 */

/**
 * 2017/12/31
 * 688. Knight Probability in Chessboard
 * 
class Solution {
	int[][] moves = {{1,-2},{1,2},{2,-1},{2,1},{-2,1},{-2,-1},{-1,-2},{-1,2}};
    public double knightProbability(int N, int K, int r, int c) {
        double[][] dp0 = new double[N][N];
        for(double[] row : dp0){Arrays.fill(row, 1);}
        for(int k = 0;k < K;k++){
        	double[][] dp1 = new double[N][N];
        	for(int i = 0;i < N;i++){
        		for(int j = 0;j < N;j++){
        			for(int[] move : moves){
        				int row = move[0] + i;
        				int col = move[1] + j;
        				if(isLegal(N, row, col))dp1[i][j] += dp0[row][col];
        			}
        		}
        	}
        	dp0 = dp1;
        }
        return dp0[r][c] / Math.pow(8, K);
    }
    
    public boolean isLegal(int N,int row,int col){
    	return row >= 0 && row < N && col >= 0 && col < N;
    }
}


 */

package exercise;

import java.util.Arrays;









