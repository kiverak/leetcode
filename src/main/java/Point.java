import java.util.Objects;

class Point {
    private double x;
    private double y;

    public Point(final double x, final double y) {
        this.x = x;
        this.y = y;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public double getDistance(Point another) {
        return Math.sqrt((this.getX() - another.getX()) * (this.getX() - another.getX())
                + (this.getY() - another.getY()) * (this.getY() - another.getY()));
    }

    @Override
    public String toString() {
        return "[" + x + "," + y + ']';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Point point = (Point) o;
        double epsilon = 0.000001d;
        return Math.abs(point.x - x) < epsilon && Math.abs(point.y - y) < epsilon;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y);
    }
}
