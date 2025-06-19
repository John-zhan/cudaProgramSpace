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

        // 3. 获取各部分利用率 (GPU, Frame Buffer, Video Engine)
        NV_GPU_DYNAMIC_PSTATES_INFO_EX pstates = {0};
        pstates.version = NV_GPU_DYNAMIC_PSTATES_INFO_EX_VER;
        status = NvAPI_GPU_GetDynamicPstatesInfoEx(hGPU, &pstates);
        if (status == NVAPI_OK) {
            // 修正 1: 循环条件应使用 pstates.domainCount
            for (NvU32 i = 0; i < pstates.domainCount; ++i) {
                // 修正 2: 检查 pstates.utilization[i].bIsPresent 标志位
                if (pstates.utilization[i].bIsPresent) {
                    // 根据 domainId 判断是哪个部分的利用率
                    switch (pstates.utilization[i].domainId) {
                        case NVAPI_GPU_UTILIZATION_DOMAIN_GPU:
                            printf("GPU Usage:     %d%%\n", pstates.utilization[i].percentage);
                            break;
                        case NVAPI_GPU_UTILIZATION_DOMAIN_FB:
                            printf("Memory Usage:  %d%%\n", pstates.utilization[i].percentage);
                            break;
                        case NVAPI_GPU_UTILIZATION_DOMAIN_VID:
                            printf("Video Usage:   %d%%\n", pstates.utilization[i].percentage);
                            break;
                        default:
                            break;
                    }
                }
            }
        } else {
            printf("Failed to get utilization.\n");
        }

        // 修正 3: 移除了有问题的功耗获取部分代码
        // 旧的 NvAPI_GPU_GetPowerStatus 接口兼容性差，已移除。

        Sleep(1000); // 等待1秒
    }

    // 卸载 NVAPI
    NvAPI_Unload();
    return 0;
}
