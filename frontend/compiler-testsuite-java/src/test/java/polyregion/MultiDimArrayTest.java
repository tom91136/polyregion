package polyregion;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;
import polyregion.javalang.OffloadRunnable;

public class MultiDimArrayTest extends BaseSuite {

  @Test
  public void readsTwoDimensional() {
    int[][] grid = {{1, 2}, {3, 4}};
    assertMatchesJdk(() -> grid[0][0] + grid[1][1]);
  }

  @Test
  public void writesTwoDimensional() {
    int[][] grid = new int[2][2];
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) grid[i][j] = i + j;
        };
    polyregion.javalang.Runtime.offload(r);
    assertArrayEquals(new int[] {0, 1}, grid[0]);
    assertArrayEquals(new int[] {1, 2}, grid[1]);
  }

  @Test
  public void innerLength() {
    int[][] grid = {{1, 2, 3}, {4}};
    assertMatchesJdk(() -> grid[0].length + grid[1].length);
  }

  @Test
  public void innerLengthsDifferPerRow() {
    int[][] grid = {{1, 2, 3}, {4}};
    assertMatchesJdk("each row reports its own length", () -> grid[0].length * 10 + grid[1].length);
  }

  @Test
  public void innerLengthAtADynamicIndex() {
    int[][] grid = {{1}, {2, 3}, {4, 5, 6}};
    assertMatchesJdk(
        () -> {
          int acc = 0;
          for (int i = 0; i < grid.length; i++) acc = acc * 10 + grid[i].length;
          return acc;
        });
  }

  @Test
  public void sumsJaggedRowsByTheirOwnLength() {
    int[][] grid = {{1, 2, 3}, {4}, {5, 6}};
    assertMatchesJdk(
        () -> {
          int acc = 0;
          for (int i = 0; i < grid.length; i++)
            for (int j = 0; j < grid[i].length; j++) acc += grid[i][j];
          return acc;
        });
  }

  @Test
  public void allocatedRowLength() {
    assertMatchesJdk(
        () -> {
          int[][] scratch = new int[2][5];
          return scratch[0].length * 10 + scratch.length;
        });
  }

  @Test
  public void rowLengthOfAnAliasedGrid() {
    int[][] grid = {{1, 2, 3, 4}, {5}};
    assertMatchesJdk(
        () -> {
          int[][] alias = grid;
          return alias[0].length * 10 + alias[1].length;
        });
  }

  @Test
  public void outerLength() {
    int[][] grid = {{1, 2}, {3, 4}, {5, 6}};
    assertMatchesJdk(() -> grid.length);
  }

  @Test
  public void jaggedRows() {
    int[][] grid = {{1}, {2, 3}, {4, 5, 6}};
    assertMatchesJdk(() -> grid[0][0] + grid[1][1] + grid[2][2]);
  }

  @Test
  public void doubleGrid() {
    double[][] grid = {{1.5, 2.5}, {3.5, 4.5}};
    assertMatchesJdk(() -> grid[0][1] + grid[1][0]);
  }

  @Test
  public void writesEveryCell() {
    int[][] grid = new int[3][2];
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 3; i++) for (int j = 0; j < 2; j++) grid[i][j] = i * 10 + j;
        };
    polyregion.javalang.Runtime.offload(r);
    assertArrayEquals(new int[] {0, 1}, grid[0]);
    assertArrayEquals(new int[] {10, 11}, grid[1]);
    assertArrayEquals(new int[] {20, 21}, grid[2]);
  }

  @Test
  public void rowAliasedIntoLocal() {
    int[][] grid = {{1, 2, 3}, {4, 5, 6}};
    assertMatchesJdk(
        () -> {
          int[] row = grid[1];
          return row[0] + row[2];
        });
  }

  @Test
  public void passesGridToHelper() {
    int[][] grid = {{1, 2}, {3, 4}};
    assertMatchesJdk(() -> total(grid, 2, 2));
  }

  static int total(int[][] g, int rows, int cols) {
    int acc = 0;
    for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) acc += g[i][j];
    return acc;
  }

  @Test
  public void threeDimensional() {
    assertMatchesJdk(
        () -> {
          int[][][] cube = new int[2][2][2];
          cube[1][0][1] = 7;
          return cube[1][0][1];
        });
  }

  @Test
  public void allocatesFromCapturedDimensions() {
    int rows = 3;
    int cols = 4;
    assertMatchesJdk(
        () -> {
          int[][] scratch = new int[rows][cols];
          scratch[2][3] = 5;
          return scratch[2][3] + scratch[0][0];
        });
  }

  @Test
  public void emptyOuterArrayReportsZeroLength() {
    int[][] grid = new int[0][];
    assertMatchesJdk(() -> grid.length);
  }

  @Test
  public void emptyOuterArrayIsNotNull() {
    int[][] grid = new int[0][];
    assertMatchesJdk("an empty grid is not a null one", () -> grid != null ? 1 : 0);
  }

  @Test
  public void allocatesTwoDimensionalInRegion() {
    assertMatchesJdk(
        () -> {
          int[][] scratch = new int[2][3];
          scratch[1][2] = 9;
          return scratch[1][2];
        });
  }
}
