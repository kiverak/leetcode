import java.util.Arrays;
import java.util.Scanner;

public class Kolya {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        String[] input = scanner.nextLine().split(" ");

        int n = Integer.parseInt(input[0]);
        int m = Integer.parseInt(input[1]);
        int q = Integer.parseInt(input[2]);

        int[][] servers = new int[n][m + 1];

        for (int i = 0; i < n; i++) {
            Arrays.fill(servers[i], 1);
            servers[i][m] = 0;
        }

        int i;
        int j;

        for (int k = 0; k < q; k++) {
            input = scanner.nextLine().split(" ");

            switch (input[0]) {
                case "RESET": {
                    i = Integer.parseInt(input[1]) - 1;
                    for (int l = 0; l < m; l++)
                        servers[i][l] = 1;
                    servers[i][m]++;
                    break;
                }
                case "DISABLE": {
                    i = Integer.parseInt(input[1]) - 1;
                    j = Integer.parseInt(input[2]) - 1;
                    servers[i][j] = 0;
                    break;
                }
                case "GETMAX": {
                    System.out.println(getMax(servers, n, m + 1));
                    break;
                }
                case "GETMIN": {
                    System.out.println(getMin(servers, n, m + 1));
                }
            }
        }
    }

    private static int getMax(int[][] servers, int n, int m) {
        int max = Integer.MIN_VALUE;
        int server = 0;

        for (int k = 0; k < n; k++) {
            int sum = 0;
            int a = 0;

            for (int l = 0; l < m - 1; l++)
                a += servers[k][l];

            sum = a * servers[k][m - 1];

            if (sum > max) {
                max = sum;
                server = k;
            }
        }

        return server;
    }

    private static int getMin(int[][] servers, int n, int m) {
        int min = Integer.MAX_VALUE;
        int server = 0;

        for (int k = 0; k < n; k++) {
            int sum = 0;
            int a = 0;

            for (int l = 0; l < m - 1; l++)
                a += servers[k][l];

            sum = a * servers[k][m - 1];

            if (sum < min) {
                min = sum;
                server = k;
            }
        }

        return server;
    }
}
