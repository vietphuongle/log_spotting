CC = g++
CFLAGS = -std=c++11 -Wall
OPENCV_LIBS = -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_calib3d -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_xfeatures2d
OPENCV_INCLUDE = -I/usr/local/include/opencv4

all: log_spotting

log_spotting: log_spotting.cpp
	$(CC) $(CFLAGS) $(OPENCV_INCLUDE) $^ -o $@ $(OPENCV_LIBS)

clean:
	rm -f log_spotting

