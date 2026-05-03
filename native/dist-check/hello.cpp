#include <cstdio>
int main() {
    int sum = 0;
    for (int i = 0; i < 10; ++i) sum += i;
    std::printf("sum=%d\n", sum);
    return sum == 45 ? 0 : 1;
}
