CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c11 -fopenmp -march=native
LDFLAGS = -lm -fopenmp

SRC_DIR = src
BIN_DIR = bin

SRCS = $(SRC_DIR)/matrix.c $(SRC_DIR)/benchmark.c $(SRC_DIR)/main.c
OBJS = $(SRCS:.c=.o)

ifeq ($(OS),Windows_NT)
    TARGET = $(BIN_DIR)/matrix_bench.exe
    RM = del /Q /F
    MKDIR = if not exist $(BIN_DIR) mkdir $(BIN_DIR)
else
    TARGET = $(BIN_DIR)/matrix_bench
    RM = rm -rf
    MKDIR = mkdir -p $(BIN_DIR)
endif

.PHONY: all clean test bench quick

all: $(TARGET)

$(TARGET): $(SRCS)
	@$(MKDIR)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

test: $(TARGET)
	$(TARGET) --test

bench: $(TARGET)
	$(TARGET) --size 1024 --tile 64 --threads 4

quick: $(TARGET)
	$(TARGET) --size 512 --tile 32 --threads 4

clean:
	@if exist bin\matrix_bench.exe del /Q /F bin\matrix_bench.exe
	@if exist bin rmdir /S /Q bin
	@if exist src\*.o del /Q /F src\*.o
