import java.util.*;

public class SolutionMedium {

    private int lo, maxLen;

    public boolean canPartition(int[] nums) {
        int sum = 0;
        int n = nums.length;
        for (int i : nums) sum += i;
        if (sum % 2 != 0) return false;
        sum /= 2;
        boolean[] dp = new boolean[sum + 1];
        dp[0] = true;

        for (int num : nums) {
            for (int j = sum; j > 0; j--) {
                if (j >= num) {
                    dp[j] = dp[j] || dp[j - num];
                }
            }
        }

        return dp[sum];
    }

    public int findNumberOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        int[] count = new int[nums.length];
        int maxLength = 0;
        for (int i = 0; i < nums.length; i++) {
            dp[i] = 1;
            count[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    if (dp[i] < dp[j] + 1) {
                        dp[i] = dp[j] + 1;
                        count[i] = count[j];
                    } else if (dp[i] == dp[j] + 1) {
                        count[i] += count[j];
                    }
                }
            }
            maxLength = Math.max(maxLength, dp[i]);
        }
        int ans = 0;
        for (int i = 0; i < dp.length; i++) {
            if (dp[i] == maxLength) {
                ans += count[i];
            }
        }
        return ans;
    }

    public int countSubstrings(String s) {
        int counter = 1;
        for (int i = 1; i < s.length(); i++) {
            counter += countSubstringsHelper(s, i, i);
            counter += countSubstringsHelper(s, i - 1, i);
        }
        return counter;
    }

    private int countSubstringsHelper(String s, int left, int right) {
        int counter = 0;
        while (left >= 0 && right < s.length()) {
            if (s.charAt(left) == s.charAt(right)) {
                counter++;
                left--;
                right++;
            } else break;
        }
        return counter;
    }

    public boolean canJump(int[] nums) {
        int cur = 0;
        while (cur < nums.length - 1) {
            int num = nums[cur];
            if (num == 0) {
                int backSteps = 1;
                while (num < backSteps) {
                    cur -= 1;
                    if (cur < 1) return false;
                    num = nums[cur];
                    backSteps++;
                }
            }
            cur += num;
        }

        return true;
    }

    public int uniquePaths(int m, int n) {
        int[] dp = new int[m];
        Arrays.fill(dp, 1);

        for (int i = 1; i < n; i++) {
            for (int j = 1; j < m; j++)
                dp[j] = dp[j - 1] + dp[j];
        }

        return dp[m - 1];
    }

    public int numDecodings(String s) {
        int dp1 = 1, dp2 = 0, len = s.length();
        for (int i = len - 1; i >= 0; i--) {
            int dp = s.charAt(i) == '0' ? 0 : dp1;
            if (i < len - 1 && (s.charAt(i) == '1' || s.charAt(i) == '2' && s.charAt(i + 1) < '7'))
                dp += dp2;
            dp2 = dp1;
            dp1 = dp;
        }
        return dp1;
    }

    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i <= target; i++)
            for (int num : nums)
                if (num <= i) dp[i] += dp[i - num];
        return dp[target];
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        Boolean[] memo = new Boolean[s.length() + 1];
        return wordBreak(s, wordDict, 0, memo);
    }

    private boolean wordBreak(String s, List<String> wordDict, int k, Boolean[] memo) {
        if (k == s.length()) {
            return true;
        }

        if (memo[k] != null) {
            return memo[k];
        }

        for (int i = 0; i < wordDict.size(); i++) {
            String word = wordDict.get(i);
            if (s.startsWith(word, k)) {
                if (wordBreak(s, wordDict, k + word.length(), memo)) return memo[k] = true;
            }
        }

        return memo[k] = false;
    }

    public String longestPalindrome(String s) {
        int len = s.length();
        if (len < 2)
            return s;

        for (int i = 0; i < len - 1; i++) {
            extendPalindrome(s, i, i);  //assume odd length, try to extend Palindrome as possible
            extendPalindrome(s, i, i + 1); //assume even length.
        }
        return s.substring(lo, lo + maxLen);
    }

    private void extendPalindrome(String s, int j, int k) {
        while (j >= 0 && k < s.length() && s.charAt(j) == s.charAt(k)) {
            j--;
            k++;
        }
        if (maxLen < k - j - 1) {
            lo = j + 1;
            maxLen = k - j - 1;
        }
    }

    public int lengthOfLIS2(int[] nums) {
        int[] tails = new int[nums.length];
        int size = 0;
        for (int num : nums) {
            int i = 0, j = size;
            while (i != j) {
                int m = (i + j) / 2;
                if (tails[m] < num)
                    i = m + 1;
                else
                    j = m;
            }
            tails[i] = num;
            if (i == size) ++size;
        }
        return size;
    }

    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        int len = 0;
        for (int i = 1; i < nums.length; i++) {
            int pos = binarySearch(dp, len, nums[i]);
            if (nums[i] < dp[pos]) dp[pos] = nums[i];
            if (pos > len) {
                len = pos;
                dp[len] = nums[i];
            }
        }
        return len + 1;
    }

    private int binarySearch(int[] dp, int len, int val) {
        int left = 0;
        int right = len;
        while (left + 1 < right) {
            int mid = left + (right - left) / 2;
            if (dp[mid] == val) {
                return mid;
            } else {
                if (dp[mid] < val) {
                    left = mid;
                } else {
                    right = mid;
                }
            }
        }
        if (dp[right] < val) return len + 1;
        else if (dp[left] >= val) return left;
        else return right;
    }

    public int maxProduct2(int[] nums) {
        int n = nums.length;
        int l = 1, r = 1;
        int ans = nums[0];

        for (int i = 0; i < n; i++) {
            //if any of l or r become 0 then update it to 1
            l = l == 0 ? 1 : l;
            r = r == 0 ? 1 : r;

            l *= nums[i];   //prefix product
            r *= nums[n - 1 - i];    //suffix product

            ans = Math.max(ans, Math.max(l, r));
        }

        return ans;
    }

    public int maxProduct(int[] nums) {
        int max = nums[0], min = nums[0], ans = nums[0];

        for (int i = 1; i < nums.length; i++) {
            int temp = max;  // store the max because before updating min your max will already be updated

            max = Math.max(Math.max(max * nums[i], min * nums[i]), nums[i]);
            min = Math.min(Math.min(temp * nums[i], min * nums[i]), nums[i]);

            if (max > ans) ans = max;
        }

        return ans;
    }

    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++)
            for (int c : coins)
                if (i >= c) dp[i] = Math.min(dp[i], dp[i - c] + 1);
        return dp[amount] > amount ? -1 : dp[amount];
    }

    public int robTwo(int[] nums) {
        if (nums.length == 1) return nums[0];
        return Math.max(robTwoHelper(nums, 0, nums.length - 2), robTwoHelper(nums, 1, nums.length - 1));
    }

    private int robTwoHelper(int[] num, int lo, int hi) {
        int include = 0, exclude = 0;
        for (int j = lo; j <= hi; j++) {
            int i = include, e = exclude;
            include = e + num[j];
            exclude = Math.max(e, i);
        }
        return Math.max(include, exclude);
    }

    public int maxSubArray(int[] nums) {
        int max = Integer.MIN_VALUE;
        int sum = 0;

        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            max = Math.max(sum, max);
            if (sum < 0) sum = 0;
        }

        return max;
    }

    public int rob(int[] nums) {
        int prev1 = 0;
        int prev2 = 0;
        int dp;

        for (final int num : nums) {
            dp = Math.max(prev1, prev2 + num);
            prev2 = prev1;
            prev1 = dp;
        }

        return prev1;
    }

    public List<String> letterCombinations(String digits) {
        List<String> list = new ArrayList<>();
        if (digits.isEmpty()) return list;
        Map<Character, Character> map = new HashMap<>();
        map.put('a', '2');
        map.put('b', '2');
        map.put('c', '2');
        map.put('d', '3');
        map.put('e', '3');
        map.put('f', '3');
        map.put('g', '4');
        map.put('h', '4');
        map.put('i', '4');
        map.put('j', '5');
        map.put('k', '5');
        map.put('l', '5');
        map.put('m', '6');
        map.put('n', '6');
        map.put('o', '6');
        map.put('p', '7');
        map.put('q', '7');
        map.put('r', '7');
        map.put('s', '7');
        map.put('t', '8');
        map.put('u', '8');
        map.put('v', '8');
        map.put('w', '9');
        map.put('x', '9');
        map.put('y', '9');
        map.put('z', '9');
        letterCombinationsHelper(list, "", digits, 0, map);
        return list;
    }

    private void letterCombinationsHelper(List<String> list, String temp, String digits, int index, Map<Character, Character> map) {
        if (index == digits.length()) {
            list.add(temp);
            return;
        }
        char digit = digits.charAt(index);
        for (Map.Entry<Character, Character> entry : map.entrySet()) {
            if (entry.getValue() == digit) {
                temp += entry.getKey();
                letterCombinationsHelper(list, temp, digits, index + 1, map);
                temp = temp.substring(0, temp.length() - 1);
            }
        }

    }

    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        List<String> path = new ArrayList<>();
        partitionHelper(0, s, path, res);
        return res;
    }

    private void partitionHelper(int index, String s, List<String> path, List<List<String>> res) {
        if (index == s.length()) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = index; i < s.length(); i++) {
            if (isPalindrome(s, index, i)) {
                path.add(s.substring(index, i + 1));
                partitionHelper(i + 1, s, path, res);
                path.remove(path.size() - 1);
            }
        }
    }

    private boolean isPalindrome(String s, int start, int end) {
        while (start <= end) {
            if (s.charAt(start++) != s.charAt(end--)) return false;
        }
        return true;
    }

    public int findTargetSumWays(int[] nums, int target) {
        if (nums == null || nums.length == 0) return 0;
        int[] counter = {0};
        findTargetSumWaysHelper(nums, target, 0, counter);
        return counter[0];
    }

    private void findTargetSumWaysHelper(int[] nums, int remain, int index, int[] counter) {
        if (index == nums.length) {
            if (remain == 0) counter[0]++;
            return;
        }

        findTargetSumWaysHelper(nums, remain + nums[index], index + 1, counter);
        findTargetSumWaysHelper(nums, remain - nums[index], index + 1, counter);
    }

    public List<String> generateParenthesis(int n) {
        List<String> list = new ArrayList<>();
        if (n == 0) return list;
        generateParenthesisHelper(list, "", n, n);
        return list;
    }

    private void generateParenthesisHelper(List<String> list, String str, int open, int close) {
        if (close == 0) {
            list.add(str);
            return;
        }
        if (open > 0) generateParenthesisHelper(list, str + "(", open - 1, close);
        if (close > open) generateParenthesisHelper(list, str + ")", open, close - 1);
    }

    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> list = new ArrayList<>();
        if (k == 0 || n == 0) return list;
        int[] nums = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        combinationSum3Helper(list, new ArrayList<>(), k, n, 1);
        return list;
    }

    private void combinationSum3Helper(List<List<Integer>> list, List<Integer> tempList, int len, int remain, int start) {
        if (remain == 0 && tempList.size() == len) list.add(new ArrayList<>(tempList));
        else if (remain <= 0) return;
        else {
            for (int i = start; i < 10; i++) {
                if (i > remain) break;
                tempList.add(i);
                combinationSum3Helper(list, tempList, len, remain - i, i + 1);
                tempList.remove(tempList.size() - 1);
            }
        }
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> list = new ArrayList<>();
        if (candidates == null || candidates.length == 0) return list;
        Arrays.sort(candidates);
        combinationSum2Helper(list, new ArrayList<>(), candidates, target, 0);
        return list;
    }

    private void combinationSum2Helper(List<List<Integer>> list, List<Integer> tempList, int[] nums, int remain, int start) {
        if (remain == 0) list.add(new ArrayList<>(tempList));
        else if (remain > 0) {
            for (int i = start; i < nums.length; i++) {
                if (nums[i] > remain) break;
                if (i > start && nums[i] == nums[i - 1]) continue;
                tempList.add(nums[i]);
                combinationSum2Helper(list, tempList, nums, remain - nums[i], i + 1);
                tempList.remove(tempList.size() - 1);
            }
        }
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> list = new ArrayList<>();
        if (candidates.length == 0) return list;
        combinationSumHelper(list, new ArrayList<>(), candidates, target, 0);
        return list;
    }

    private void combinationSumHelper(List<List<Integer>> list, List<Integer> tempList, int[] nums, int remain, int start) {
        if (remain < 0) return;
        else if (remain == 0) list.add(new ArrayList<>(tempList));
        else {
            for (int i = start; i < nums.length; i++) {
                tempList.add(nums[i]);
                combinationSumHelper(list, tempList, nums, remain - nums[i], i); // not i + 1 because we can reuse same elements
                tempList.remove(tempList.size() - 1);
            }
        }
    }

    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> combs = new ArrayList<List<Integer>>();
        combine(combs, new ArrayList<Integer>(), 1, n, k);
        return combs;
    }

    public static void combine(List<List<Integer>> combs, List<Integer> comb, int start, int n, int k) {
        if (k == 0) {
            combs.add(new ArrayList<Integer>(comb));
            return;
        }
        for (int i = start; i <= n - k + 1; i++) {
            comb.add(i);
            combine(combs, comb, i + 1, n, k - 1);
            comb.remove(comb.size() - 1);
        }
    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        if (nums == null || nums.length == 0) return list;
        permuteUniqueHelper(list, nums, 0);
        return list;
    }

    private void permuteUniqueHelper(List<List<Integer>> list, int[] nums, int ind) {
        List<Integer> integerList = new ArrayList<>(nums.length);
        for (int i : nums) integerList.add(i);
        list.add(integerList);
        for (int i = ind; i < nums.length; i++) {
            Set<Integer> set = new HashSet<>();
            set.add(nums[i]);
            for (int j = i + 1; j < nums.length; j++) {
                if (!set.contains(nums[j])) {
                    int temp1 = nums[i];
                    int temp2 = nums[j];
                    nums[i] = temp2;
                    nums[j] = temp1;
                    permuteUniqueHelper(list, nums, i + 1);
                    nums[i] = temp1;
                    nums[j] = temp2;
                    set.add(nums[j]);
                }
            }
        }
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        if (nums == null || nums.length == 0) return list;
        permuteHelper(list, nums, 0);
        return list;
    }

    private void permuteHelper(List<List<Integer>> list, int[] nums, int ind) {
        List<Integer> integerList = new ArrayList<>(nums.length);
        for (int i : nums) integerList.add(i);
        list.add(integerList);

        for (int i = ind; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                int temp1 = nums[i];
                int temp2 = nums[j];
                nums[i] = temp2;
                nums[j] = temp1;
                permuteHelper(list, nums, i + 1);
                nums[i] = temp1;
                nums[j] = temp2;
            }
        }
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> numList = new ArrayList<>();
        result.add(new ArrayList<>());
        backtrackWithDup(0, nums, numList, result, true);
        return result;
    }

    private void backtrackWithDup(int offset, int[] nums, List<Integer> numList, List<List<Integer>> result, boolean isPicked) {
        // base case
        if (offset >= nums.length) {
            return;
        }
        int val = nums[offset];
        // duplicate checking (convert && to ||)
        if (offset == 0 || nums[offset - 1] != nums[offset] || isPicked) {
            // pick
            numList.add(val);
            backtrackWithDup(offset + 1, nums, numList, result, true);
            result.add(new ArrayList<>(numList));  // add to the result list
            numList.remove(numList.size() - 1);
        }
        // not pick
        backtrackWithDup(offset + 1, nums, numList, result, false);
    }


    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        backtrack(list, new ArrayList<>(), nums, 0);
        return list;
    }

    private void backtrack(List<List<Integer>> list, List<Integer> tempList, int[] nums, int start) {
        list.add(new ArrayList<>(tempList));
        for (int i = start; i < nums.length; i++) {
            tempList.add(nums[i]);
            backtrack(list, tempList, nums, i + 1);
            tempList.remove(tempList.size() - 1);
        }
    }

    public List<String> letterCasePermutation(String s) {
        return letterCasePermutationHelper(s.toCharArray(), 0);
    }

    private List<String> letterCasePermutationHelper(char[] chars, int index) {
        List<String> list = new ArrayList<>();
        if (chars.length == 0 || index == chars.length) return list;
        char ch = chars[index];
        if (!Character.isLetter(ch)) {
            if (index == chars.length - 1) {
                list.add(new String(chars));
            } else {
                list.addAll(letterCasePermutationHelper(chars, index + 1));
            }
        } else {
            char ch1 = Character.toUpperCase(ch);
            char ch2 = Character.toLowerCase(ch);

            if (index == chars.length - 1) {
                chars[index] = ch1;
                list.add(new String(chars));
                chars[index] = ch2;
                list.add(new String(chars));
            } else {
                chars[index] = ch1;
                list.addAll(letterCasePermutationHelper(chars, index + 1));
                chars[index] = ch2;
                list.addAll(letterCasePermutationHelper(chars, index + 1));
            }
        }

        return list;
    }

    public int longestConsecutive(int[] nums) {
        int max = 0;

        Set<Integer> set = new HashSet<Integer>();
        for (int i = 0; i < nums.length; i++) {
            set.add(nums[i]);
        }

        for (int i = 0; i < nums.length; i++) {
            int count = 1;

            // look left
            int num = nums[i];
            while (set.contains(--num)) {
                count++;
                set.remove(num);
            }

            // look right
            num = nums[i];
            while (set.contains(++num)) {
                count++;
                set.remove(num);
            }

            max = Math.max(max, count);
        }

        return max;
    }

    public boolean exist(char[][] board, String word) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == word.charAt(0)) {
                    if (search(board, word, i, j, 0)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private boolean search(char[][] board, String word, int i, int j, int index) {
        if (index == word.length()) {
            return true;
        }
        if (i >= board.length || i < 0 || j >= board[i].length || j < 0 || board[i][j] != word.charAt(index)) {
            return false;
        }
        if (board[i][j] == '#') {
            return false;
        }
        char tmp = board[i][j];
        board[i][j] = '#';

        boolean found = search(board, word, i - 1, j, index + 1) ||
                search(board, word, i + 1, j, index + 1) ||
                search(board, word, i, j - 1, index + 1) ||
                search(board, word, i, j + 1, index + 1);

        board[i][j] = tmp;
        return found;
    }

    public void rotate(int[][] matrix) {
        int n = matrix.length;

        for (int i = 0; i < n / 2; i++) {
            for (int j = i; j < n - 1 - i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][i];
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];
                matrix[j][n - 1 - i] = temp;
            }
        }
    }

    public List<Integer> spiralOrder(int[][] matrix) {
        int mTop = 0, mBot = matrix.length - 1;
        int nLeft = 0, nRight = matrix[0].length - 1;
        List<Integer> list = new ArrayList<>();

        while (mTop <= mBot && nLeft <= nRight) {
            for (int i = nLeft; i <= nRight; i++) {
                list.add(matrix[mTop][i]);
            }
            mTop++;

            for (int i = mTop; i <= mBot; i++) {
                list.add(matrix[i][nRight]);
            }
            nRight--;

            if (mTop <= mBot)
                for (int i = nRight; i >= nLeft; i--) {
                    list.add(matrix[mBot][i]);
                }
            mBot--;

            if (nLeft <= nRight)
                for (int i = mBot; i >= mTop; i--) {
                    list.add(matrix[i][nLeft]);
                }
            nLeft++;
        }

        return list;
    }

    public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        boolean isRow0 = false, isCol0 = false;

        for (int j = 0; j < n; j++) {
            if (matrix[0][j] == 0)
                isRow0 = true;
        }

        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0)
                isCol0 = true;
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[0][j] == 0 || matrix[i][0] == 0)
                    matrix[i][j] = 0;
            }
        }

        if (isRow0) {
            for (int j = 0; j < n; j++)
                matrix[0][j] = 0;
        }

        if (isCol0) {
            for (int i = 0; i < m; i++)
                matrix[i][0] = 0;
        }
    }

    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> duplicates = new ArrayList<>();
        int n;
        for (int i = 0; i < nums.length; i++) {
            n = Math.abs(nums[i]);
            if (nums[n - 1] < 0) {
                duplicates.add(n);
            } else {
                nums[n - 1] *= -1;
            }
        }
        return duplicates;
    }

    public int findDuplicate(int[] nums) {
        int slow = 0, fast = 0;
        slow = nums[slow];
        fast = nums[nums[fast]];
        while (slow != fast) {
            if (slow == nums[slow])
                return slow;
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        fast = 0;
        while (slow != fast) {
            if (slow == nums[slow])
                return slow;
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        int curr = 1;
        for (int i = 0; i < n; i++) {
            ans[i] = curr;
            curr *= nums[i];
        }
        curr = 1;
        for (int i = n - 1; i >= 0; i--) {
            ans[i] *= curr;
            curr *= nums[i];
        }
        return ans;
    }

    public ListNode deleteMiddle(ListNode head) {
        if (head.next == null) return null;
        ListNode slow = head;
        ListNode fast = head;
        ListNode prev = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            prev = slow;
            slow = slow.next;
        }
        prev.next = slow.next;
        return head;
    }

    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        rightView(root, result, 0);
        return result;
    }

    public void rightView(TreeNode current, List<Integer> list, int depth) {
        if (current == null) return;
        if (depth == list.size()) {
            list.add(current.val);
        }

        rightView(current.right, list, depth + 1);
        rightView(current.left, list, depth + 1);
    }

    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        boolean[] keys = new boolean[rooms.size()];
        canVisitAllRoomsHelper(0, keys, rooms);
        for (boolean key : keys) {
            if (!key) return false;
        }
        return true;
    }

    private void canVisitAllRoomsHelper(int curr, boolean[] keys, List<List<Integer>> rooms) {
        keys[curr] = true;
        for (Integer next : rooms.get(curr)) {
            if (!keys[next]) {
                canVisitAllRoomsHelper(next, keys, rooms);
            }
        }
    }

    public boolean canVisitAllRooms2(List<List<Integer>> rooms) {
        boolean[] visited = new boolean[rooms.size()];
        Queue<Integer> queue = new ArrayDeque<>();
        queue.add(0);

        while (queue.size() > 0) {
            int curr = queue.remove();
            if (visited[curr]) continue;
            visited[curr] = true;

            for (int key : rooms.get(curr)) {
                if (!visited[key]) {
                    queue.add(key);
                }
            }
        }

        for (boolean room : visited) {
            if (!room) return false;
        }
        return true;
    }

    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        int maxLength = 0;
        int[] charIndex = new int[128];
        Arrays.fill(charIndex, -1);
        int left = 0;

        for (int right = 0; right < n; right++) {
            if (charIndex[s.charAt(right)] >= left) {
                left = charIndex[s.charAt(right)] + 1;
            }
            charIndex[s.charAt(right)] = right;
            maxLength = Math.max(maxLength, right - left + 1);
        }

        return maxLength;
    }

    public int[] dailyTemperatures(int[] temperatures) {
        int[] results = new int[temperatures.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < temperatures.length; i++) {
            while (!stack.isEmpty() && temperatures[stack.peek()] < temperatures[i]) {
                results[stack.peek()] = i - stack.pop();
            }
            stack.push(i);
        }

        return results;
    }

    public int[] dailyTemperatures2(int[] temperatures) {
        int[] res = new int[temperatures.length];
        Deque<Integer> deque = new ArrayDeque<>();

        for (int i = temperatures.length - 1; i >= 0; --i) {
            if (deque.isEmpty()) {
                deque.offerFirst(i);
                res[i] = 0;
            } else {
                while (!deque.isEmpty() && temperatures[i] >= temperatures[deque.peekFirst()]) {
                    deque.pollFirst();
                }

                if (deque.isEmpty()) {
                    res[i] = 0;
                } else {
                    res[i] = deque.peekFirst() - i;
                }

                deque.offerFirst(i);
            }
        }

        return res;
    }

    public static int[] dailyTemperatures3(int[] temperatures) {
        int[] result = new int[temperatures.length];
        int[] stack = new int[temperatures.length];
        int top = -1;
        for(int i = 0; i < temperatures.length; i++) {
            while(top > -1 && temperatures[i] > temperatures[stack[top]]) {
                result[stack[top]] = i - stack[top];
                top--;
            }
            stack[++top] = i;
        }
        return result;
    }

}
