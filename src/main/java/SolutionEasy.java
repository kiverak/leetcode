import java.math.BigInteger;
import java.util.*;

class SolutionEasy {
    int maxValue = Integer.MIN_VALUE;

    public double findMaxAverage(int[] nums, int k) {
        int sum = 0;
        for (int i = 0; i < k; i++) sum += nums[i];
        int max = sum;
        for (int i = k; i < nums.length; i++) {
            sum = sum + nums[i] - nums[i - k + 1];
            if (sum > max) max = sum;
        }

        return (double) max / k;
    }

    public boolean isSubsequence(String s, String t) {
        if (s.length() == 0) return true;
        if (s.length() > t.length()) return false;

        int ind = 0;

        for (char ch : t.toCharArray()) {
            if (ch == s.charAt(ind)) ind++;
            if (ind == s.length()) return true;
        }

        return false;
    }

    public int[][] construct2DArray(int[] original, int m, int n) {
        if (original.length != m * n) return new int[0][0];

        int[][] array = new int[m][n];

        for (int i = 0; i < m; i++) {
            System.arraycopy(original, i * n, array[i], 0, n);
        }

        return array;
    }

    public boolean backspaceCompare(String s, String t) {
        int i = s.length() - 1;
        int j = t.length() - 1;

        while (i >= 0 || j >= 0) {
            int i1 = getNextValidIndex(s, i);
            int j1 = getNextValidIndex(t, j);

            if (i1 < 0 && j1 < 0) return true;

            if (i1 < 0 || j1 < 0 || s.charAt(i1) != t.charAt(j1)) return false;

            i = i1 - 1;
            j = j1 - 1;
        }

        return true;
    }

    private int getNextValidIndex(String str, int index) {
        int backSpaceCount = 0;

        while (index >= 0) {
            if (str.charAt(index) == '#') backSpaceCount++;
            else if (backSpaceCount > 0) backSpaceCount--;
            else break;

            index--;
        }

        return index;
    }

    public int[] sortedSquares(int[] nums) {
        if (nums.length == 1) return new int[]{nums[0] * nums[0]};
        int[] res = new int[nums.length];

        int left = 0;
        int right = nums.length - 1;

        for (int i = res.length - 1; i > -1; i--) {
            if (nums[right] > -nums[left]) {
                res[i] = nums[right] * nums[right];
                right--;
            } else {
                res[i] = nums[left] * nums[left];
                left++;
            }
        }

        return res;
    }

    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int diff = target - nums[i];
            if (map.containsKey(diff)) {
                return new int[]{map.get(diff), i};
            }
            map.put(nums[i], i);
        }
        return null;
    }

    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null) return false;

        return isSubtreeHelper(root, subRoot)
                || isSubtree(root.left, subRoot)
                || isSubtree(root.right, subRoot);
    }

    private boolean isSubtreeHelper(TreeNode root, TreeNode subRoot) {
        if (root == null && subRoot == null) return true;
        if (root == null || subRoot == null) return false;

        return (root.val == subRoot.val)
                && isSubtreeHelper(root.left, subRoot.left)
                && isSubtreeHelper(root.right, subRoot.right);
    }

    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) return null;
        if (root1 == null) return root2;
        if (root2 == null) return root1;

        TreeNode root = new TreeNode(root1.val + root2.val);
        root.left = mergeTrees(root1.left, root2.left);
        root.right = mergeTrees(root1.right, root2.right);

        return root;
    }

    public int diameterOfBinaryTree(TreeNode root) {
        height(root);
        return maxValue - 1;
    }

    private int height(TreeNode root) {
        if (root == null) return 0;
        int left = height(root.left);
        int right = height(root.right);
        maxValue = Math.max(maxValue, left + right + 1);
        return Math.max(left, right) + 1;
    }

    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> result = new ArrayList<>();

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int levelSize = queue.size();

            double levelSum = 0.0;

            for (int i = 0; i < levelSize; i++) {
                TreeNode currentNode = queue.poll();

                // add the node's value to the running sum
                levelSum += currentNode.val;

                if (currentNode.left != null) queue.offer(currentNode.left);
                if (currentNode.right != null) queue.offer(currentNode.right);
            }

            // append the current level's average to the result array
            result.add(levelSum / levelSize);
        }
        return result;
    }

    public char nextGreatestLetter(char[] letters, char target) {
        char nextGreatestLetter = letters[0];
        int left = 0;
        int right = letters.length - 1;
        int mid;

        while (left <= right) {
            mid = left + (right - left) / 2;
            if (letters[mid] <= target) left = mid + 1;
            else {
                nextGreatestLetter = letters[mid];
                right = mid - 1;
            }
        }

        return nextGreatestLetter;
    }

    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        int mid;

        while (left <= right) {
            mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] > target) right = mid - 1;
            else left = mid + 1;
        }

        return -1;
    }

    public ListNode middleNode(ListNode head) {
        if (head.next == null) return head;

        ListNode slow = head;
        ListNode fast = head;

        while (fast.next != null) {
            slow = slow.next;
            if (fast.next.next == null) {
                fast = fast.next;
            } else {
                fast = fast.next.next;
            }
        }

        return slow;
    }

    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> list = new ArrayList<>();
        int idx = -1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < 0) {
                idx = nums[i] * -1 - 1;
            } else {
                idx = nums[i] - 1;
            }

            if (nums[idx] > 0) {
                nums[idx] = -nums[idx];
            }

        }

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                list.add(i + 1);
            }
        }

        return list;
    }

    public char findTheDifference(String s, String t) {
        char c = 0;
        for (char cs : s.toCharArray()) c ^= cs;
        for (char ct : t.toCharArray()) c ^= ct;
        return c;
    }

    public int firstUniqChar(String s) {
        int ans = Integer.MAX_VALUE;
        for (char c = 'a'; c <= 'z'; c++) {
            int index = s.indexOf(c);
            if (index != -1 && index == s.lastIndexOf(c)) {
                ans = Math.min(ans, index);
            }
        }

        return ans == Integer.MAX_VALUE ? -1 : ans;
    }

    public boolean canConstruct(String ransomNote, String magazine) {
        if (ransomNote.length() > magazine.length()) return false;
        int[] alphabets_counter = new int[26];

        for (char c : magazine.toCharArray())
            alphabets_counter[c - 'a']++;

        for (char c : ransomNote.toCharArray()) {
            if (alphabets_counter[c - 'a'] == 0) return false;
            alphabets_counter[c - 'a']--;
        }
        return true;
    }

    public int guessNumber(int n) {
        int left = 1;
        int right = n;
        int mid;
        while (left < right) {
            mid = left + (right - left) / 2;
            int res = guess(mid);
            if (res == 0) return mid;
            else if (res < 0) right = mid - 1;
            else left = mid + 1;
        }

        return left;
    }

    int guess(int num) {
        return 0;
    }

    public boolean isPerfectSquare(int num) {
        if (46340 * 46340 < num) return false;

        int left = 1;
        int right = 46340;
        int mid;
        int sqrt;

        while (left <= right) {
            mid = left + (right - left) / 2;
            sqrt = mid * mid;
            if (sqrt == num) {
                return true;
            } else if (sqrt < num) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return false;
    }

    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums1) {
            set.add(num);
        }

        Set<Integer> intersec = new HashSet<>();
        for (int num : nums2) {
            if (set.contains(num)) {
                intersec.add(num);
            }
        }

        int[] intersecArr = new int[intersec.size()];
        int j = 0;
        for (int num : intersec) {
            intersecArr[j] = num;
            j++;
        }

        return intersecArr;
    }

    public String reverseVowels(String s) {
        String vowels = "aeiouAEIOU";
        int first = 0, last = s.length() - 1;
        char[] array = s.toCharArray();
        while (first < last) {
            while (first < last && vowels.indexOf(array[first]) == -1) {
                first++;
            }
            while (first < last && vowels.indexOf(array[last]) == -1) {
                last--;
            }
            char temp = array[first];
            array[first] = array[last];
            array[last] = temp;
            first++;
            last--;
        }
        return new String(array);
    }

    public void reverseString(char[] s) {
        char temp;
        for (int i = 0, j = s.length - 1; i < j; i++, j--) {
            temp = s[i];
            s[i] = s[j];
            s[j] = temp;
        }
    }

    public boolean isPowerOfFour(int n) {
        return (n > 0) && (n & (n - 1)) == 0 && (n & 0x55555555) != 0;
    }

    public int[] countBits(int n) {
        int[] res = new int[n + 1];
        res[0] = 0;

        for (int i = 1; i <= n; i++) {
            if ((i & 1) == 0) {
                res[i] = res[i >> 1];
            } else {
                res[i] = res[i - 1] + 1;
            }
        }

        return res;
    }

    public boolean isPowerOfThree(int n) {
        if (n <= 0) return false;
        if (n == 1) return true;

        int x = 1;
        for (int i = 1; i < 20; i++) {
            x *= 3;
            if (n == x) return true;
        }
        return false;
    }

    public boolean canWinNim(int n) {
        return (n % 4 != 0);
    }

    public boolean wordPattern(String pattern, String s) {
        String[] stringArr = s.split(" ");

        if (pattern.length() != stringArr.length) return false;

        Map<Character, String> map = new HashMap<>();
        for (int i = 0; i < pattern.length(); i++) {
            if (map.containsKey(pattern.charAt(i))) {
                if (!map.get(pattern.charAt(i)).equals(stringArr[i])) return false;
            } else {
                if (map.containsValue(stringArr[i])) return false;
                map.put(pattern.charAt(i), stringArr[i]);
            }
        }

        return true;
    }

    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length == 0) return;

        int insertPos = 0;
        for (int num : nums) {
            if (num != 0) nums[insertPos++] = num;
        }

        while (insertPos < nums.length) {
            nums[insertPos++] = 0;
        }
    }

    public int firstBadVersion(int n) {
        int left = 1;
        int right = n;

        while (left < right) {
            int mid = left + (right - left) / 2;

            if (isBadVersion(mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        return left;
    }

    private boolean isBadVersion(int version) {
        return version >= 4;
    }

    public int missingNumber(int[] nums) {
        int sum = nums.length;
        for (int i = 0; i < nums.length; i++) sum -= nums[i] - i;
        return sum;
    }

    public boolean isUgly(int n) {
        if (n <= 0) return false;
        while (n % 2 == 0) n /= 2;
        while (n % 3 == 0) n /= 3;
        while (n % 5 == 0) n /= 5;
        return n == 1;
    }

    public int addDigits(int num) {
        if (num == 0) return 0;
        return num % 9 == 0 ? 9 : num % 9;
    }

    public List<String> binaryTreePaths(TreeNode root) {
        List<String> list = new ArrayList<>();
        if (root != null) binaryTreePathsHelper(list, root, "");
        return list;
    }

    private void binaryTreePathsHelper(List<String> list, TreeNode root, String str) {
        str += String.valueOf(root.val);

        if (root.left == null && root.right == null) {
            list.add(str);
        } else {
            if (root.left != null) binaryTreePathsHelper(list, root.left, str + "->");
            if (root.right != null) binaryTreePathsHelper(list, root.right, str + "->");
        }
    }

    public boolean isAnagram(String s, String t) {
        int[] alphabet = new int[26];
        for (int i = 0; i < s.length(); i++) alphabet[s.charAt(i) - 'a']++;
        for (int i = 0; i < t.length(); i++) alphabet[t.charAt(i) - 'a']--;
        for (int i : alphabet) if (i != 0) return false;
        return true;
    }

    public boolean isPalindrome(ListNode head) {
        if (head == null) return false;

        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        ListNode prev = slow;
        slow = slow.next;
        prev.next = null;
        ListNode temp;
        while (slow != null) {
            temp = prev;
            prev = slow;
            slow = slow.next;
            prev.next = temp;
        }

        fast = head;
        slow = prev;
        while (slow != null) {
            if (fast.val != slow.val) return false;
            fast = fast.next;
            slow = slow.next;
        }

        return true;
    }

    public boolean isPowerOfTwo(int n) {
//        return n>0 && Integer.bitCount(n) == 1;
        if (n == 1) return true;
        if (n <= 0 || n % 2 == 1) return false;

        return isPowerOfTwo(n / 2);
    }

    public List<String> summaryRanges(int[] nums) {
        List<String> list = new ArrayList<>();
        if (nums.length == 0) return list;

        int i = 0;
        int first;
        while (i != nums.length) {
            first = nums[i];
            while (i < nums.length - 1 && nums[i] + 1 == nums[i + 1])
                i++;
            if (first == nums[i])
                list.add(String.valueOf(first));
            else list.add(String.valueOf(first).concat("->").concat(String.valueOf(nums[i])));
            i++;
        }

        return list;
    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;

        TreeNode temp = root.left;

        root.left = invertTree(root.right);
        root.right = invertTree(temp);

        return root;
    }

    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            if (i > k) set.remove(nums[i - k - 1]);
            if (!set.add(nums[i])) return true;
        }
        return false;
    }

    public ListNode reverseList(ListNode head) {
        ListNode next;
        ListNode newHead = null;
        while (head != null) {
            next = head.next;
            head.next = newHead;
            newHead = head;
            head = next;
        }
        return newHead;
    }

    public boolean isIsomorphic(String s, String t) {
        int[] map1 = new int[200];
        int[] map2 = new int[200];

        for (int i = 0; i < s.length(); i++) {
            if (map1[s.charAt(i)] != map2[t.charAt(i)])
                return false;

            map1[s.charAt(i)] = i + 1;
            map2[t.charAt(i)] = i + 1;
        }
        return true;
    }

    public ListNode removeElements(ListNode head, int val) {
        if (head == null) return null;
        head.next = removeElements(head.next, val);
        return head.val == val ? head.next : head;
    }

    public boolean isHappy(int n) {
        int s = n;
        int f = n; // slow , fast

        do {
            s = sumOfSquaresOfDigits(s); // slow computes only once
            f = sumOfSquaresOfDigits(sumOfSquaresOfDigits(f)); // fast computes 2 times

            if (s == 1) return true; // if we found 1 then happy indeed !!!
        } while (s != f); // else at some point they will meet in the cycle

        return false;
    }

    private int sumOfSquaresOfDigits(int n) {
        int sum = 0;
        while (n != 0) {
            sum += (n % 10) * (n % 10);
            n = n / 10;
        }

        return sum;
    }

    public int hammingWeight2(int n) {
        int setBitCount = 0;
        while (n != 0) {
            n &= (n - 1); // to clear the right most set bit
            ++setBitCount;
        }
        return setBitCount;
    }

    public int hammingWeight(int n) {
        int ones = 0;
        while (n != 0) {
            ones = ones + (n & 1);
            n = n >>> 1;
        }
        return ones;
    }

    public int reverseBits(int n) {
        int res = 0;
        for (int i = 0; i < 32; i++) {
            res = (res << 1) | (n & 1);
            n = n >> 1;
        }
        return res;
    }

    public int titleToNumber(String columnTitle) {
        int num = 0;

        for (char ch : columnTitle.toUpperCase().toCharArray()) {
            num *= 26;
            num += ch - 'A' + 1;
        }

        return num;
    }

    public String convertToTitle(int columnNumber) {
        Map<Integer, Character> letters = new HashMap<>();
        char ch = 'A';
        for (int i = 1; i < 27; i++) {
            letters.put(i, ch);
            ch++;
        }
        letters.put(0, 'Z');

        StringBuilder sb = new StringBuilder();

        if (columnNumber < 27) {
            sb.append(letters.get(columnNumber));
        } else {
            while (columnNumber != 0) {
                sb.append(letters.get(columnNumber % 26));
                columnNumber = (columnNumber - 1) / 26;
            }
        }

        return sb.reverse().toString();
    }

    public int majorityElement(int[] nums) {
        int count = 0, ret = 0;
        for (int num : nums) {
            if (count == 0)
                ret = num;
            if (num != ret)
                count--;
            else
                count++;
        }
        return ret;
    }


    public void quickSort(int[] arr, int begin, int end) {
        if (begin < end) {
            int partitionIndex = partition(arr, begin, end);

            quickSort(arr, begin, partitionIndex - 1);
            quickSort(arr, partitionIndex + 1, end);
        }
    }

    private int partition(int[] arr, int begin, int end) {
        int pivot = arr[end];
        int i = (begin - 1);

        for (int j = begin; j < end; j++) {
            if (arr[j] <= pivot) {
                i++;

                int swapTemp = arr[i];
                arr[i] = arr[j];
                arr[j] = swapTemp;
            }
        }

        int swapTemp = arr[i + 1];
        arr[i + 1] = arr[end];
        arr[end] = swapTemp;

        return i + 1;
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;

        ListNode a = headA;
        ListNode b = headB;

        //if a & b have different len, then we will stop the loop after second iteration
        while (a != b) {
            //for the end of first iteration, we just reset the pointer to the head of another linkedlist
            a = a == null ? headB : a.next;
            b = b == null ? headA : b.next;
        }

        return a;
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if (root == null) return list;

        list.addAll(postorderTraversal(root.left));
        list.addAll(postorderTraversal(root.right));

        list.add(root.val);

        return list;
    }

    private void postorderTraversalHelper(TreeNode root, List<Integer> list) {

    }

    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();

        if (root != null) {
            list.add(root.val);
            list.addAll(preorderTraversal(root.left));
            list.addAll(preorderTraversal(root.right));
        }

        return list;
    }

    public boolean hasCycle(ListNode head) {
        Set<ListNode> nodeSet = new HashSet<>();
        nodeSet.add(head);
        ListNode current = head;
        while (current != null) {
            if (nodeSet.contains(current.next)) return true;
            nodeSet.add(current.next);
            current = current.next;
        }

        return false;
    }

    public boolean isPalindrome(String s) {
        if (s == null || s.isEmpty()) return true;

        char[] chars = s.toCharArray();

        int left = 0;
        int right = chars.length - 1;

        char leftCh;
        char rightCh;

        while (left < right) {
            leftCh = chars[left];
            rightCh = chars[right];

            if (!Character.isLetterOrDigit(leftCh)) {
                left++;
            } else if (!Character.isLetterOrDigit(rightCh)) {
                right--;
            } else {
                if (Character.toLowerCase(leftCh) != Character.toLowerCase(rightCh)) return false;
                left++;
                right--;
            }
        }

        return true;
    }

    public int maxProfit(int[] prices) {
        int max = 0, min = prices[0];
        for (int i = 1; i < prices.length; i++) {

            if (min < prices[i])
                max = Math.max(prices[i] - min, max);
            else
                min = prices[i];
        }
        return max;
    }

    public int maxProfitTrading(int[] prices) {
        if (prices.length == 0 || prices.length == 1) return 0;

        int sum = 0;
        int buyPrice = 0;
        boolean inMarket = false;

        for (int i = 0; i < prices.length; i++) {
            if (inMarket) {
                if (i != prices.length - 1) {
                    if (prices[i + 1] < prices[i]) {
                        sum += prices[i] - buyPrice;
                        inMarket = false;
                    }
                } else {
                    sum += prices[i] - buyPrice;
                }

            } else {
                if (i != prices.length - 1) {
                    if (prices[i + 1] > prices[i]) {
                        buyPrice = prices[i];
                        inMarket = true;
                    }
                }
            }
        }

        return sum;
    }

    public List<Integer> getRow(int rowIndex) {
        List<Integer> row = Collections.singletonList(1);

        for (int i = 1; i <= rowIndex; i++) {
            List<Integer> curRow = new ArrayList<>();
            curRow.add(1);
            for (int j = 1; j < i; j++) {
                curRow.add(row.get(j - 1) + row.get(j));
            }
            curRow.add(1);
            row = curRow;
        }

        return row;
    }

    public List<List<Integer>> pascalTriangle(int numRows) {
        if (numRows == 0) return null;

        List<List<Integer>> list = new ArrayList<>();

        list.add(Collections.singletonList(1));

        for (int i = 1; i < numRows; i++) {
            List<Integer> row = new ArrayList<>();
            row.add(1);
            for (int j = 1; j < i; j++) {
                row.add(list.get(i - 1).get(j - 1) + list.get(i - 1).get(j));
            }
            row.add(1);
            list.add(row);
        }

        return list;
    }

    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;

        if (root.left == null && root.right == null)
            return root.val == targetSum;

        return hasPathSum(root.left, targetSum - root.val)
                || hasPathSum(root.right, targetSum - root.val);
    }

    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int level = 1;
        while (!queue.isEmpty()) {
            for (int i = 0; i < queue.size(); i++) {
                TreeNode curNode = queue.poll();
                if (curNode.left == null && curNode.right == null) {
                    return level;
                }
                if (curNode.left != null) {
                    queue.offer(curNode.left);
                }
                if (curNode.right != null) {
                    queue.offer(curNode.right);
                }
            }
            level++;
        }
        return level;
    }

    boolean balanced = true;

    public boolean isBalanced(TreeNode root) {
        if (root == null) return true;
        maxDepthComparing(root);
        return balanced;
    }

    private int maxDepthComparing(TreeNode root) {
        if (root == null || !balanced) return 0;

        int left = maxDepthComparing(root.left);
        int right = maxDepthComparing(root.right);

        if (Math.abs(left - right) > 1)
            balanced = false;

        return Math.max(left, right) + 1;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBSTHelper(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBSTHelper(int[] nums, int left, int right) {
        if (left > right) return null;

        int mid = (left + right) / 2;

        TreeNode root = new TreeNode(nums[mid]);

        root.left = sortedArrayToBSTHelper(nums, left, mid - 1);
        root.right = sortedArrayToBSTHelper(nums, mid + 1, right);

        return root;
    }

    public int maxDepth(TreeNode root) {
        if (root == null) return 0;

        int kLeft = maxDepth(root.left);
        int kRight = maxDepth(root.right);

        return kLeft > kRight ? kLeft + 1 : kRight + 1;
    }

    public boolean isSymmetric(TreeNode root) {
        if (root == null)
            return true;
        return isSymmetricSubtree(root.left, root.right);
    }

    private boolean isSymmetricSubtree(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        if (left == null || right == null) return false;
        if (left.val != right.val) return false;
        return isSymmetricSubtree(left.left, right.right) && isSymmetricSubtree(left.right, right.left);
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null)
            return true;

        if ((p == null && q != null) || (p != null && q == null))
            return false;

        if (p.val != q.val)
            return false;

        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        // method 1: recursion

        inorderTraversalHelper(root, res);
        return res;
    }

    //helper function for method 1
    private void inorderTraversalHelper(TreeNode root, List<Integer> res) {
        if (root != null) {
            inorderTraversalHelper(root.left, res);
            res.add(root.val);
            inorderTraversalHelper(root.right, res);
        }
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        if (n == 0)
            return;

        if (m == 0) {
            System.arraycopy(nums2, 0, nums1, 0, nums1.length);
            return;
        }

        int i = m - 1;
        int j = n - 1;
        int k = nums1.length - 1;

        while (j >= 0) {
            if (i >= 0 && nums1[i] >= nums2[j]) {
                nums1[k] = nums1[i];
                nums1[i] = 0;
                k--;
                i--;
            } else {
                nums1[k] = nums2[j];
                k--;
                j--;
            }
        }
    }

    public void merge2(int[] nums1, int m, int[] nums2, int n) {
        if (nums1.length - m >= 0)
            System.arraycopy(nums2, 0, nums1, m, nums1.length - m);

        quickSort(nums1, 0, nums1.length - 1);
    }

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode list = head;
        while (list.next != null) {
            if (list.val == list.next.val)
                list.next = list.next.next;
            else
                list = list.next;
        }

        return head;
    }

    public int climbStairs(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        if (n == 2) return 2;

        int n1 = 1;
        int n2 = 2;
        int sum = 0;

        for (int i = 2; i < n; i++) {
            sum = n1 + n2;
            n1 = n2;
            n2 = sum;
        }

        return sum;
    }

    public int mySqrt(int x) {
        if (x == 0)
            return 0;

        int start = 1;
        int end = x;

        while (start < end) {
            int mid = start / 2 + end / 2;

            if (mid <= x / mid && (mid + 1) > x / (mid + 1))
                return mid;
            else if (mid > x / mid)
                end = mid;
            else
                start = mid + 1;
        }

        return start;
    }

    public String addBinary(String a, String b) {
        StringBuilder res = new StringBuilder();
        int i = a.length() - 1;
        int j = b.length() - 1;
        int carry = 0;
        while (i >= 0 || j >= 0) {
            int sum = carry;
            if (i >= 0) sum += a.charAt(i--) - '0';
            if (j >= 0) sum += b.charAt(j--) - '0';
            carry = sum > 1 ? 1 : 0;
            res.append(sum % 2);
        }
        if (carry != 0) res.append(carry);
        return res.reverse().toString();
    }

    public int lengthOfLastWord(String s) {
        String[] strArr = s.split(" ");
        return strArr[strArr.length - 1].length();
    }

    public int searchInsert(int[] nums, int target) {
        if (target > nums[nums.length - 1]) {
            return nums.length;
        }

        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {

            int mid = (left + right) / 2;

            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return left;
    }

    public int strStr(String haystack, String needle) {
        int len = haystack.length() - needle.length();

        for (int i = 0; i < len + 1; i++) {
            if (haystack.startsWith(needle, i))
                return i;
        }

        return -1;
    }

    public int removeElement(int[] nums, int val) {
        int i = 0;
        for (int j = 0; j < nums.length; j++) {
            if (nums[j] != val) {
                nums[i] = nums[j];
                i++;
            }
        }
        return i;
    }

    public boolean isValid(String s) {
        if (s.length() % 2 != 0)
            return false;

        String[] strSplit = s.split("");

        List<String> arrayString = new ArrayList<>(Arrays.asList(strSplit));

        for (int i = 0; i < arrayString.size(); i++) {
            switch (arrayString.get(i)) {
                case "}": {
                    if (i == 0 || !arrayString.get(i - 1).equals("{"))
                        return false;
                    else {
                        arrayString.remove(i - 1);
                        arrayString.remove(i - 1);
                        i -= 2;
                    }
                    break;
                }
                case "]": {
                    if (i == 0 || !arrayString.get(i - 1).equals("["))
                        return false;
                    else {
                        arrayString.remove(i - 1);
                        arrayString.remove(i - 1);
                        i -= 2;
                    }
                    break;
                }
                case ")": {
                    if (i == 0 || !arrayString.get(i - 1).equals("("))
                        return false;
                    else {
                        arrayString.remove(i - 1);
                        arrayString.remove(i - 1);
                        i -= 2;
                    }
                }
            }

            if (i + 1 == arrayString.size())
                break;
        }
        if (arrayString.size() == 0)
            return true;

        return false;
    }

    //    s contains only the characters ('I', 'V', 'X', 'L', 'C', 'D', 'M').
//    It is guaranteed that s is a valid roman numeral in the range [1, 3999]
    public int romanToInt(String s) {
        int num = 0;

        char[] chars = s.toCharArray();

        for (int i = 0; i < chars.length; i++) {
            switch (chars[i]) {
                case 'M': {
                    num += 1000;
                    break;
                }
                case 'D': {
                    num += 500;
                    break;
                }
                case 'C': {
                    if (i != chars.length - 1 && chars[i + 1] == 'M') {
                        num -= 100;
                        break;
                    } else if (i != chars.length - 1 && chars[i + 1] == 'D') {
                        num -= 100;
                        break;
                    } else {
                        num += 100;
                        break;
                    }
                }
                case 'L': {
                    num += 50;
                    break;
                }
                case 'X': {
                    if (i != chars.length - 1 && chars[i + 1] == 'L') {
                        num -= 10;
                        break;
                    } else if (i != chars.length - 1 && chars[i + 1] == 'C') {
                        num -= 10;
                        break;
                    } else {
                        num += 10;
                        break;
                    }
                }
                case 'V': {
                    num += 5;
                    break;
                }
                case 'I': {
                    if (i != chars.length - 1 && chars[i + 1] == 'V') {
                        num -= 1;
                        break;
                    } else if (i != chars.length - 1 && chars[i + 1] == 'X') {
                        num -= 1;
                        break;
                    } else {
                        num += 1;
                        break;
                    }
                }
            }
        }

        return num;
    }

    public boolean isPalindrome(int x) {
        if (x == 0 || x == 1)
            return true;

        if (x < 0)
            return false;

        char[] strX = String.valueOf(x).toCharArray();

        for (int i = 0, j = strX.length - 1; j - i >= 1; i++, j--) {
            if (strX[i] != strX[j])
                return false;
        }

        return true;
    }

    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null) {
            return list2;
        } else if (list2 == null) {
            return list1;
        } else if (list1.val < list2.val) {
            list1.next = mergeTwoLists(list1.next, list2);
            return list1;
        } else {
            list2.next = mergeTwoLists(list1, list2.next);
            return list2;
        }
    }

    public int[] intersect(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map = new HashMap<>();
        List<Integer> list = new ArrayList<>();
        for (int num : nums1) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        for (int num : nums2) {
            if (map.containsKey(num)) {
                list.add(num);
                map.put(num, map.get(num) - 1);
                if (map.get(num) == 0) {
                    map.remove(num);
                }
            }
        }
        int[] res = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return res;
    }

    public String toCamelCase(String s) {

        String[] strings = s.split("[-_]");

        StringBuilder sb = new StringBuilder(strings[0]);

        for (int i = 1; i < strings.length; i++) {
            char[] chars = strings[i].toCharArray();
            chars[0] = Character.toUpperCase(chars[0]);
            sb.append(chars);
        }

        return sb.toString();
    }

    public String longestCommonPrefix(String[] strs) {
        StringBuilder sb = new StringBuilder();
        boolean stop = false;

        for (int i = 0; i < strs[0].length(); i++) {
            Character ch = strs[0].charAt(i);
            for (String str : strs) {
                if (i == str.length() || ch != str.charAt(i)) {
                    stop = true;
                    break;
                }
            }
            if (stop) {
                break;
            } else {
                sb.append(ch);
            }
        }

        return sb.toString();
    }

    public int singleNumber(int[] nums) {
        int result = 0;
        for (int i : nums) {
            result ^= i;
            System.out.println("i = " + i + ", result = " + result);
        }
        return result;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        return getListNodeFromInt(getNumberFromListNode(l1).add(getNumberFromListNode(l2)).toString());
    }

    public BigInteger getNumberFromListNode(ListNode l) {
        StringBuilder sb = new StringBuilder();
        ListNode next = l;
        while (next != null) {
            ListNode current = next;
            next = current.next;
            sb.append(current.val);
        }

        return new BigInteger(sb.reverse().toString());
    }

    public ListNode getListNodeFromInt(String num) {
        char[] chars = new StringBuilder(num).reverse().toString().toCharArray();

        ListNode current = new ListNode(Integer.parseInt(String.valueOf(chars[0])));
        ListNode listNode = current;

        for (int i = 1; i < chars.length; i++) {
            ListNode next = new ListNode(Integer.parseInt(String.valueOf(chars[i])));
            current.next = next;
            current = next;
        }

        return listNode;
    }

    public int countNodes(TreeNode root) {
        if (root == null) return 0;
        return 1 + countNodes(root.left) + countNodes(root.right);
    }

    public int thirdMax(int[] nums) {
        int[] arr = {nums[0], nums[0], nums[0]};

        for (int num : nums) {
            if (num == arr[0] || num == arr[1] || num == arr[2]) continue;
            if (num > arr[2]) {
                arr[0] = arr[1];
                arr[1] = arr[2];
                arr[2] = num;
            } else if (num > arr[1] || arr[1] == arr[2]) {
                if (arr[1] == arr[2]) {
                    arr[1] = num;
                    arr[0] = num;
                } else {
                    arr[0] = arr[1];
                    arr[1] = num;
                }
            } else if (num > arr[0] || arr[0] == arr[1]) {
                arr[0] = num;
            }
        }

        if (nums.length < 3 || arr[1] == arr[2] || arr[0] == arr[1]) return arr[2];
        return arr[0];
    }

    public int thirdMax2(int[] nums) {
        Integer max1 = null;
        Integer max2 = null;
        Integer max3 = null;

        for (Integer num : nums) {
            if (num.equals(max1) || num.equals(max2) || num.equals(max3)) continue;
            if (max3 == null || num > max3) {
                max1 = max2;
                max2 = max3;
                max3 = num;
            } else if (max2 == null || num > max2) {
                max1 = max2;
                max2 = num;
            } else if (max1 == null || num > max1) {
                max1 = num;
            }
        }

        return max1 == null ? max3 : max1;
    }

    public int thirdMax3(int[] nums) {
        long max1 = Long.MIN_VALUE, max2 = Long.MIN_VALUE, max3 = Long.MIN_VALUE;

        for (int num : nums) {
            if (num > max3) {
                max1 = max2;
                max2 = max3;
                max3 = num;
            } else if (num > max2 && num != max3) {
                max1 = max2;
                max2 = num;
            } else if (num > max1 && num != max2 && num != max3) {
                max1 = num;
            }
        }

        return max1 == Long.MIN_VALUE ? (int) max3 : (int) max1;
    }

    public int longestPalindrome(String s) {
        int[] chars = new int[58];
        int len = 0;
        for (char ch : s.toCharArray()) chars[ch - 'A']++;
        for (int data : chars) len += data % 2 == 0 ? data : data - 1;
        return len == s.length() ? len : len + 1;
    }

    public int longestPalindrome2(String s) {
        Set<Character> hs = new HashSet<>();
        int count = 0;
        for (char c : s.toCharArray()) {
            if (hs.contains(c)) {
                hs.remove(c);
                count += 2;
            } else {
                hs.add(c);
            }
        }
        if (!hs.isEmpty()) return count + 1;
        return count;
    }

    public List<String> fizzBuzz(int n) {
        List<String> answer = new ArrayList<>(n);
        for (int i = 1; i <= n; i++) {
            if (i % 3 == 0 && i % 5 == 0) {
                answer.add("FizzBuzz");
            } else if (i % 3 == 0) {
                answer.add("Fizz");
            } else if (i % 5 == 0) {
                answer.add("Buzz");
            } else {
                answer.add(String.valueOf(i));
            }
        }
        return answer;
    }

    public String mergeAlternately(String word1, String word2) {
        char[] merge = new char[word1.length() + word2.length()];
        int minLen = word1.length() < word2.length() ? word1.length() : word2.length();

        char[] wordCharArray = word1.toCharArray();
        for (int i = 0; i < minLen; i++) {
            merge[i * 2] = wordCharArray[i];
        }

        wordCharArray = word2.toCharArray();
        for (int i = 0; i < minLen; i++) {
            merge[i * 2 + 1] = wordCharArray[i];
        }

        wordCharArray = word1.length() > word2.length() ? word1.toCharArray() : word2.toCharArray();
        System.arraycopy(wordCharArray, minLen * 2 - minLen, merge, minLen * 2, merge.length - minLen * 2);

        return new String(merge);
    }

    public String gcdOfStrings(String str1, String str2) {
        String sStr;
        String lStr;
        if (str1.length() < str2.length()) {
            sStr = str1;
            lStr = str2;
        } else {
            sStr = str2;
            lStr = str1;
        }

        for (int i = sStr.length(); i > 0; i--) {
            String gcd = sStr.substring(0, i);
            if (checkGcd(sStr, gcd) && checkGcd(lStr, gcd)) {
                return gcd;
            }
        }

        return "";
    }

    private boolean checkGcd(String str, String gcd) {
        str = str.replaceAll(gcd, "");
        return str.isEmpty();
    }

    public String gcdOfStrings2(String str1, String str2) {
        String sStr;
        String lStr;
        if (str1.length() < str2.length()) {
            sStr = str1;
            lStr = str2;
        } else {
            sStr = str2;
            lStr = str1;
        }

        if (lStr.startsWith(sStr)) {
            lStr = lStr.substring(sStr.length());
            if (lStr.length() == 0) {
                return sStr;
            }
            return gcdOfStrings2(sStr, lStr);
        } else {
            return "";
        }
    }

    public int largestAltitude(int[] gain) {
        int max = 0;
        int current = 0;
        for (int j : gain) {
            current += j;
            if (current > max) max = current;
        }
        return max;
    }

    public List<List<Integer>> findDifference(int[] nums1, int[] nums2) {
        Set<Integer> set1 = new HashSet<>();
        Set<Integer> set2 = new HashSet<>();

        for (int num : nums1) {
            set1.add(num);
        }
        for (int num : nums2) {
            set2.add(num);
            set1.remove(num);
        }
        for (int num : nums1) {
            set2.remove(num);
        }

        List<List<Integer>> answer = new ArrayList<>();
        answer.add(new ArrayList<>(set1));
        answer.add(new ArrayList<>(set2));

        return answer;
    }

    public String removeStars(String s) {
        Stack<Character> stack = new Stack<>();
        char[] charArray = s.toCharArray();
        int count = 0;
        for (int i = charArray.length - 1; i >= 0; i--) {
            char ch = charArray[i];
            if (ch == '*') {
                count++;
            } else {
                if (count > 0) {
                    count--;
                } else {
                    stack.add(ch);
                }
            }
        }

        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }

        return sb.toString();
    }

    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();

        findLeafs(root1, list1);
        findLeafs(root2, list2);

        return list1.equals(list2);
    }

    private void findLeafs(TreeNode root, List<Integer> list) {
        if (root == null) return;
        if (root.left == null && root.right == null) {
            list.add(root.val);
            return;
        }
        findLeafs(root.left, list);
        findLeafs(root.right, list);
    }

    public TreeNode searchBST(TreeNode root, int val) {
        if (val == root.val) return root;
        if (root.left != null) {
            TreeNode left = searchBST(root.left, val);
            if (left != null && val == left.val) return left;
        }
        if (root.right != null) {
            TreeNode right = searchBST(root.right, val);
            if (right != null && val == right.val) return right;
        }
        return null;
    }
}
