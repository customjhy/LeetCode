package exercise;

import java.util.Scanner;

public class Test {

	public static void main(String[] args) {
		Solution solu = new Solution();
		Scanner input = new Scanner(System.in);
		int[][] matrix = new int[3][3];
		for(int i = 0;i < 3;i++){
			matrix[0][i] = i;
			matrix[1][i] = 3 + i;
			matrix[2][i] = 6 + i;
		}
		System.out.println(matrix);
		System.out.println(matrix);
	}
}