#include <iostream>
#include "mylib.h"

int main() {
    int result = add(3, 5);
    std::cout << "Result from DLL: " << result << std::endl;
    return 0;
}
