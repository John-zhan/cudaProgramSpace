#include <iostream>
#include <windows.h>
typedef int(*AddFunc)(int, int);

int main() {
    HMODULE h = LoadLibraryA("mylib.dll");
    if (h) {
        AddFunc add = (AddFunc)GetProcAddress(h, "add");
        if (add) {
            printf("Result: %d\n", add(4, 7));
        }
        FreeLibrary(h);
    }
}
