#include <algorithm>
#include <cstdio>
#include <execution>
#include <vector>
int main() {
    std::vector<int> v(10);
    for (int i = 0; i < 10; ++i) v[i] = i;
    std::for_each(std::execution::par_unseq, v.begin(), v.end(), [](int &x) { x *= 2; });
    int sum = 0;
    for (int x : v) sum += x;
    std::printf("sum=%d\n", sum);
    return sum == 90 ? 0 : 1;
}
