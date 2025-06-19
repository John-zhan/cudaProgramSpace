#include <windows.h>
#include <stdio.h>
#include <nvapi.h>

void printNvStatus(NvAPI_Status status) {
    NvAPI_ShortString desc;
    NvAPI_GetErrorMessage(status, desc);
    printf("NVAPI Error: %s\n", desc);
}

int main() {
    NvAPI_Status status;

    status = NvAPI_Initialize();
    if (status != NVAPI_OK) {
        printNvStatus(status);
        return -1;
    }

    NvPhysicalGpuHandle gpuHandles[NVAPI_MAX_PHYSICAL_GPUS] = {0};
    NvU32 gpuCount = 0;
    status = NvAPI_EnumPhysicalGPUs(gpuHandles, &gpuCount);
    if (status != NVAPI_OK || gpuCount == 0) {
        printNvStatus(status);
        return -1;
    }

    NvPhysicalGpuHandle hGPU = gpuHandles[0];
    NvU32 temp = 0, fanRpm = 0, power = 0;

    while (true) {
        system("cls"); // Clear screen

        // 获取温度
        NV_GPU_THERMAL_SETTINGS thermal = {0};
        thermal.version = NV_GPU_THERMAL_SETTINGS_VER;
        status = NvAPI_GPU_GetThermalSettings(hGPU, 0, &thermal);
        if (status == NVAPI_OK) {
            printf("Temperature: %d °C\n", thermal.sensor[0].currentTemp);
        }

        // 获取风扇转速
        status = NvAPI_GPU_GetTachReading(hGPU, &fanRpm);
        if (status == NVAPI_OK) {
            printf("Fan Speed: %d RPM\n", fanRpm);
        }

        // 获取功率
        NV_GPU_DYNAMIC_PSTATES_INFO_EX pstates = {0};
        pstates.version = NV_GPU_DYNAMIC_PSTATES_INFO_EX_VER;
        status = NvAPI_GPU_GetDynamicPstatesInfoEx(hGPU, &pstates);
        if (status == NVAPI_OK) {
            for (int i = 0; i < pstates.utilization.domainCount; ++i) {
                if (pstates.utilization.domain[i].percent >= 0)
                    printf("Domain %d Utilization: %d%%\n", i, pstates.utilization.domain[i].percent);
            }
        }

        // 获取功耗（如支持）
        NV_GPU_POWER_STATUS powerStatus = {0};
        powerStatus.version = NV_GPU_POWER_STATUS_VER;
        status = NvAPI_GPU_GetPowerStatus(hGPU, &powerStatus);
        if (status == NVAPI_OK) {
            printf("Power Draw: %d mW\n", powerStatus.powerConsumption);
        }

        Sleep(1000);
    }

    NvAPI_Unload();
    return 0;
}
