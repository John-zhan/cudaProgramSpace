# 编译器和链接器 (使用 cl.exe 驱动)
CC=cl

# --- 目录设置 ---
SRC_DIR     = src
LIB_DIR     = mylib
OBJ_DIR     = obj
BIN_DIR     = bin

# --- 编译和链接选项 ---
# /I$(LIB_DIR) 添加头文件搜索路径
# /EHsc 启用 C++ 异常处理
# /MD 使用多线程动态链接运行时库
# /nologo 不显示编译器版权信息
CFLAGS      = /nologo /EHsc /MD /I$(LIB_DIR)

# /LD 创建动态链接库
DLL_FLAGS   = /LD

# --- 源文件 ---
MAIN_SRC    = $(SRC_DIR)/main.cpp
LIB_SRC     = $(LIB_DIR)/mylib.cpp
LIB_HEADER  = $(LIB_DIR)/mylib.h

# --- 中间目标文件 ---
MAIN_OBJ    = $(OBJ_DIR)/main.obj
LIB_OBJ     = $(OBJ_DIR)/mylib.obj

# --- 最终目标文件 ---
TARGET_EXE  = $(BIN_DIR)/main.exe
TARGET_DLL  = $(BIN_DIR)/mylib.dll
TARGET_LIB  = $(BIN_DIR)/mylib.lib

# --- 伪目标 ---
.PHONY: all clean dirs

# 默认目标：构建可执行文件
all: $(TARGET_EXE)

# 规则：链接可执行文件
# 依赖于 main.obj 和 mylib.lib
$(TARGET_EXE): $(MAIN_OBJ) $(TARGET_LIB)
	@echo Linking $@...
	$(CC) $(CFLAGS) $^ /link /OUT:$@

# 规则：创建 DLL 和其导入库 .lib
# 依赖于 mylib.obj
# 使用 /IMPLIB 指定 .lib 文件的输出位置
$(TARGET_DLL) $(TARGET_LIB): $(LIB_OBJ) | $(BIN_DIR)
	@echo Creating DLL and Import Library...
	$(CC) $(CFLAGS) $(DLL_FLAGS) $(LIB_OBJ) /link /OUT:$(TARGET_DLL) /IMPLIB:$(TARGET_LIB)

# 规则：编译 main.cpp
# 依赖于 main.cpp 和 mylib.h
$(MAIN_OBJ): $(MAIN_SRC) $(LIB_HEADER) | $(OBJ_DIR)
	@echo Compiling $<...
	$(CC) $(CFLAGS) /c $< /Fo$@

# 规则：编译 mylib.cpp
# 依赖于 mylib.cpp 和 mylib.h
$(LIB_OBJ): $(LIB_SRC) $(LIB_HEADER) | $(OBJ_DIR)
	@echo Compiling $<...
	$(CC) $(CFLAGS) /c $< /Fo$@

# 规则：创建输出目录 (如果不存在)
# | $(DIR) 语法表示这是一个“仅顺序”依赖，目录只需要在规则执行前存在即可
dirs: $(OBJ_DIR) $(BIN_DIR)

$(OBJ_DIR):
	@if not exist $@ mkdir $@

$(BIN_DIR):
	@if not exist $@ mkdir $@

# 清理所有生成的文件和目录
clean:
	@echo Cleaning up build artifacts...
	@if exist $(OBJ_DIR) rmdir /s /q $(OBJ_DIR)
	@if exist $(BIN_DIR) rmdir /s /q $(BIN_DIR)
	@del /Q /F *.exp 2>nul