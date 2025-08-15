import java.io.*;
import java.util.HashMap;
import java.util.Map;

public class CodeRunEasy {
    public static void uniqueNumbers() throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(System.out));

        int n = Integer.parseInt(reader.readLine());

        String[] numString = reader.readLine().split(" ");

        Map<Integer, Integer> map = new HashMap<>();
        int num;
        for (String s : numString) {
            num = Integer.parseInt(s);
            if (map.containsKey(num)) {
                map.put(num, map.get(num) + 1);
            } else {
                map.put(num, 1);
            }
        }

        int counter = 0;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (entry.getValue() == 1) counter++;
        }

        writer.write(String.valueOf(counter));

        reader.close();
        writer.close();
    }
}
