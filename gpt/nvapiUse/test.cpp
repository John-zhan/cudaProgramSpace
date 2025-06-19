#include <Windows.h>
#include <nvapi.h>
#include <stdio.h>

#pragma comment(lib, "nvapi64.lib")

int main() {
    NvAPI_Status status = NvAPI_Initialize();
    if (status == NVAPI_OK) {
        printf("NVAPI 初始化成功\n");
    } else {
        printf("NVAPI 初始化失败\n");
    }
    return 0;
}
