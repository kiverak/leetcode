import java.util.Stack;

class MyQueue {

    Stack<Integer> inputStack;
    Stack<Integer> outputStack;

    public MyQueue() {
        this.inputStack = new Stack<>();
        this.outputStack = new Stack<>();
    }

    public void push(int x) {
        inputStack.push(x);
    }

    public int pop() {
        if (outputStack.empty()) {
            while (!inputStack.empty())
                outputStack.push(inputStack.pop());
        }

        return outputStack.pop();
    }

    public int peek() {
        if (outputStack.empty()) {
            while (!inputStack.empty())
                outputStack.push(inputStack.pop());
        }

        return outputStack.peek();
    }

    public boolean empty() {
        return inputStack.empty() && outputStack.empty();
    }
}