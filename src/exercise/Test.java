package exercise;

import java.util.Scanner;
import other.WordDictionary;

public class Test {

	public static void main(String[] args) {
		Solution solu = new Solution();
		Scanner input = new Scanner(System.in);
		int[] a = new int[]{1,3,5,4};
		int[] b = new int[]{1,2,3,7};
		System.out.println(solu.minSwap(a, b));
/*		String[] a = new String[4];
		a[0] = "abd";
		a[1] = "aba";
		a[2] = "adcc";
		a[3] = "abc";
		*/
	}
}