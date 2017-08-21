HOME = ..
PROTO_PATH= .
CXX = g++
RELFLAGS = -g
CXXFLAGS += -I/usr/local/include -I. -std=c++11 

GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

.PHONY: all translate-proto
all: translate-proto

translate-proto:
	python -m grpc.tools.protoc -I. --python_out=. --grpc_python_out=. $(PROTO_PATH)/translate.proto
	cp $(PROTO_PATH)/translate*pb* $(HOME)/

clean:
	rm -f *_pb2.py *pb2_grpc.py
	rm -f $(HOME)/translate*pb*

