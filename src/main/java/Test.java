import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Test {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String input = br.readLine();
        int n = Integer.parseInt(input);
        input = br.readLine();
        int left = Integer.parseInt(input);
        input = br.readLine();
        int right = Integer.parseInt(input);

        System.out.println(countTwoAntsSteps(n, left, right));
    }

    private static int countTwoAntsSteps(int n, int left, int right) {
        int counter = 0;
        boolean lMove = false;
        boolean rMove = true;

        while (left >= 0 && right >= 0 && left <= n && right <= n) {
            if (left == right + 1) lMove = true;
            if (!lMove) left--;
            else left++;

            if (right == left - 1) rMove = false;
            if (rMove) right++;
            else right--;

            counter++;
        }

        return counter;
    }
}
