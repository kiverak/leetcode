class Main {
    public static void main(String[] args) {
        SolutionMedium solutionMedium = new SolutionMedium();
        SolutionEasy solutionEasy = new SolutionEasy();

        int[] nums = {1,5,11,5};
        String word1 = "TAUXXTAUXXTAUXXTAUXXTAUXX";
        String word2 = "TAUXXTAUXXTAUXXTAUXXTAUXXTAUXXTAUXXTAUXXTAUXX";
        int[] gain = {-5,1,5,0,-7};

        System.out.println(solutionEasy.largestAltitude(gain));
    }

    private static void printTreeNode(TreeNode listNode) {
        if (listNode == null)
            return;
        System.out.println(listNode.val);
        printTreeNode(listNode.left);
        printTreeNode(listNode.right);

    }

    private static void printListNode(ListNode listNode) {
        ListNode next = listNode;
        while (next != null) {
            System.out.print(next.val);
            next = next.next;
        }
    }

    public static void printArray(int[] array) {
        System.out.print("[");
        for (int i = 0; i < array.length; i++) {
            if (i == array.length - 1) {
                System.out.print(array[i]);
            } else {
                System.out.print(array[i] + ", ");
            }
        }
        System.out.println("]");
    }
}