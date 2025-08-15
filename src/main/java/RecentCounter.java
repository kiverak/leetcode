import java.util.LinkedList;
import java.util.Queue;

public class RecentCounter {
    private final Queue<Integer> requests;
    private final int PERIOD = 3000;

    public RecentCounter() {
        requests = new LinkedList<>();
    }

    public int ping(int t) {
        requests.add(t);
        int count = 0;
        while (requests.peek() < t - PERIOD) {
            requests.remove();
        }
        return requests.size();
    }
}
