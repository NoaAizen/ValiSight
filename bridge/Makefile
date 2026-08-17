# ValiSight N6 <-> Jetson bridge
CC ?= gcc
all: bridge_rx bridge_rx_asan
bridge_rx: bridge_rx.c bridge_protocol.h
	$(CC) -O2 -Wall -Wextra -o $@ bridge_rx.c -lz
bridge_rx_asan: bridge_rx.c bridge_protocol.h
	$(CC) -O1 -g -fsanitize=address,undefined -o $@ bridge_rx.c -lz
test: all
	python3 test_bridge_protocol.py
	python3 test_n6_bridge_tx.py
	python3 test_bridge_rx_pty.py
	python3 test_imu_holds.py
clean:
	rm -f bridge_rx bridge_rx_asan
.PHONY: all test clean
