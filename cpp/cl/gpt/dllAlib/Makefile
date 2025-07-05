# 编译器和链接器 (使用 cl.exe 驱动)
CC=cl

# --- 目录设置 ---
SRC_DIR     = src
LIB_DIR     = mylib
OBJ_DIR     = obj
BIN_DIR     = bin

# --- 编译和链接选项 ---
CFLAGS      = /nologo /EHsc /MD /I$(LIB_DIR)
DLL_FLAGS   = /LD
LDFLAGS     = /LIBPATH:$(BIN_DIR)

# --- 文件名 ---
MAIN_SRC    = $(SRC_DIR)/main.cpp
LIB_SRC     = $(LIB_DIR)/mylib.cpp
LIB_HEADER  = $(LIB_DIR)/mylib.h
MAIN_OBJ    = $(OBJ_DIR)/main.obj
LIB_OBJ     = $(OBJ_DIR)/mylib.obj
TARGET_EXE  = $(BIN_DIR)/main.exe
TARGET_DLL  = $(BIN_DIR)/mylib.dll
TARGET_LIB  = $(BIN_DIR)/mylib.lib
LIB_BASENAME = mylib.lib

# --- 伪目标 ---
# FIXED: 将 'clean' 重命名为 'cleanup'
.PHONY: all cleanup

# 默认目标
all: $(TARGET_EXE)
	@echo.
	@echo Build process finished successfully. All artifacts are in the '$(BIN_DIR)' directory.

# --- 链接规则 ---
$(TARGET_EXE): $(MAIN_OBJ) $(TARGET_LIB)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	@echo Linking $@...
	$(CC) $(CFLAGS) $(MAIN_OBJ) $(LIB_BASENAME) /link $(LDFLAGS) /OUT:$@

# --- DLL/LIB 生成规则 ---
$(TARGET_LIB): $(TARGET_DLL)

$(TARGET_DLL): $(LIB_OBJ)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	@echo Creating DLL and Import Library...
	$(CC) $(CFLAGS) $(DLL_FLAGS) $(LIB_OBJ) /link /OUT:$(TARGET_DLL) /IMPLIB:$(TARGET_LIB)

# --- 编译规则 ---
$(MAIN_OBJ): $(MAIN_SRC) $(LIB_HEADER)
	@if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
	@echo Compiling $(MAIN_SRC)...
	$(CC) $(CFLAGS) /c $(MAIN_SRC) /Fo$@

$(LIB_OBJ): $(LIB_SRC) $(LIB_HEADER)
	@if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
	@echo Compiling $(LIB_SRC)...
	$(CC) $(CFLAGS) /c $(LIB_SRC) /Fo$@

# --- 清理规则 ---
# FIXED: 将 'clean' 目标重命名为 'cleanup'
cleanup:
	@echo Cleaning up build artifacts...
	@if exist $(OBJ_DIR) rmdir /s /q $(OBJ_DIR) 2>nul
	@if exist $(BIN_DIR) rmdir /s /q $(BIN_DIR) 2>nul
	@del /Q /F *.exp 2>nul