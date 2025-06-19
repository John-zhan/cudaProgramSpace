#include <windows.h>
#include <stdio.h>
#include <nvapi.h>

// 用于打印 NVAPI 错误信息的辅助函数
void printNvStatus(NvAPI_Status status) {
    NvAPI_ShortString desc;
    NvAPI_GetErrorMessage(status, desc);
    printf("NVAPI Error: %s\n", desc);
}

int main() {
    NvAPI_Status status;

    // 初始化 NVAPI
    status = NvAPI_Initialize();
    if (status != NVAPI_OK) {
        printNvStatus(status);
        return -1;
    }

    // 枚举物理 GPU
    NvPhysicalGpuHandle gpuHandles[NVAPI_MAX_PHYSICAL_GPUS] = {0};
    NvU32 gpuCount = 0;
    status = NvAPI_EnumPhysicalGPUs(gpuHandles, &gpuCount);
    if (status != NVAPI_OK || gpuCount == 0) {
        printNvStatus(status);
        printf("No NVIDIA GPUs found.\n");
        NvAPI_Unload(); // 退出前别忘了卸载
        return -1;
    }

    // 我们只监控第一块显卡
    NvPhysicalGpuHandle hGPU = gpuHandles[0];
    NvU32 fanRpm = 0;

    while (true) {
        system("cls"); // 清屏 (Windows specific)

        // 1. 获取温度
        NV_GPU_THERMAL_SETTINGS thermal = {0};
        thermal.version = NV_GPU_THERMAL_SETTINGS_VER;
        status = NvAPI_GPU_GetThermalSettings(hGPU, 0, &thermal); // 0 表示默认传感器
        if (status == NVAPI_OK) {
            printf("Temperature: %d C\n", thermal.sensor[0].currentTemp);
        } else {
            printf("Failed to get temperature.\n");
        }

        // 2. 获取风扇转速 (Tachometer Reading)
        status = NvAPI_GPU_GetTachReading(hGPU, &fanRpm);
        if (status == NVAPI_OK) {
            printf("Fan Speed:   %d RPM\n", fanRpm);
        } else {
            printf("Failed to get fan speed (or fan is not spinning).\n");
        }

        // 3. 获取各部分利用率 (适用于旧版 NVAPI)
        NV_GPU_DYNAMIC_PSTATES_INFO_EX pstates = {0};
        pstates.version = NV_GPU_DYNAMIC_PSTATES_INFO_EX_VER;
        status = NvAPI_GPU_GetDynamicPstatesInfoEx(hGPU, &pstates);
        if (status == NVAPI_OK) {
            // 在旧版 NVAPI 中，utilization 数组的索引是固定的
            // 索引 0: GPU 利用率
            if (pstates.utilization[0].bIsPresent) {
                printf("GPU Usage:     %d%%\n", pstates.utilization[0].percentage);
            }
            // 索引 1: 显存利用率
            if (pstates.utilization[1].bIsPresent) {
                printf("Memory Usage:  %d%%\n", pstates.utilization[1].percentage);
            }
            // 索引 2: 视频引擎利用率
            if (pstates.utilization[2].bIsPresent) {
                printf("Video Usage:   %d%%\n", pstates.utilization[2].percentage);
            }
        } else {
            printf("Failed to get utilization.\n");
        }
        
        Sleep(1000); // 等待1秒
    }

    // 卸载 NVAPI
    NvAPI_Unload();
    return 0;
}