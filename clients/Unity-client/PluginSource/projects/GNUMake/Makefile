SRCDIR = ../../src
INCLUDEDIR = ../../include
DEPSDIR = ../../dependencies
SRCS = $(SRCDIR)/RenderingPlugin.cpp \
$(SRCDIR)/RenderAPI.cpp \
$(SRCDIR)/RenderAPI_OpenGLCoreES.cpp \
$(SRCDIR)/RenderAPI_Vulkan.cpp \
$(DEPSDIR)/gl3w/gl3w.c
OBJS = ${SRCS:.cpp=.o}
OBJS := $(OBJS:.c=.o)
UNITY_DEFINES = -DSUPPORT_OPENGL_UNIFIED=1 -DSUPPORT_VULKAN=1 -DUNITY_LINUX=1
INCLUDE_DIRS = -I$(INCLUDEDIR) -I$(DEPSDIR)
CXXFLAGS = $(UNITY_DEFINES) -O2 -fPIC $(INCLUDE_DIRS)
CFLAGS = $(UNITY_DEFINES) -O2 -fPIC $(INCLUDE_DIRS)
LDFLAGS = -shared -rdynamic
LIBS = -lGL
PLUGIN_SHARED = libRenderingPlugin.so
CXX ?= g++
CC ?= gcc

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<

all: shared

clean:
	rm -f $(OBJS) $(PLUGIN_SHARED)

shared: $(OBJS)
	$(CXX) $(LDFLAGS) -o $(PLUGIN_SHARED) $(OBJS) $(LIBS)
