VERSION=		0.5
#CUDA=			/cm/shared/package/cuda100/toolkit/10.0.130
CUDA=			$(shell dirname `dirname \`which nvcc\``)
#CUDA=			/usr/local/cuda
#POWER_SENSOR=		$(HOME)/projects/libpowersensor-master/build
ARCH=			$(shell arch)
CC=			gcc
CXX=			g++ #-Wno-deprecated-declarations
NVCC=			nvcc
INCLUDES=		-I.
INCLUDES+=		-I$(CUDA)/include
#INCLUDES+=		-I$(POWER_SENSOR)/include
CXXFLAGS+=		-std=c++11 -O3 -g -fpic -fopenmp $(INCLUDES) -DNDEBUG
NVCCFLAGS=		$(INCLUDES)

#CXXFLAGS+=		-march=core-avx2 -mcmodel=medium

LIBTCC_SOURCES=		util/cu.cc\
			util/nvrtc.cc\
			libtcc/CorrelatorKernel.cc\
			libtcc/Correlator.cc\
			libtcc/Kernel.cc


CORRELATOR_TEST_SOURCES=test/CorrelatorTest/CorrelatorTest.cc\
			test/CorrelatorTest/Options.cc\
			test/Common/Record.cc\
			test/Common/UnitTest.cc

OPENCL_TEST_SOURCES=	test/OpenCLCorrelatorTest/OpenCLCorrelatorTest.cc

SIMPLE_EXAMPLE_SOURCES=	test/SimpleExample/SimpleExample.cu


LIBTCC_OBJECTS=		$(LIBTCC_SOURCES:%.cc=%.o) libtcc/TCCorrelator.o
SIMPLE_EXAMPLE_OBJECTS=	$(SIMPLE_EXAMPLE_SOURCES:%.cu=%.o)
CORRELATOR_TEST_OBJECTS=$(CORRELATOR_TEST_SOURCES:%.cc=%.o)
#OPENCL_TEST_OBJECTS=	$(OPENCL_TEST_SOURCES:%.cc=%.o)

OBJECTS=		$(LIBTCC_OBJECTS)\
			$(SIMPLE_EXAMPLE_OBJECTS)\
			$(CORRELATOR_TEST_OBJECTS)\
			$(OPENCL_TEST_OBJECTS)

SHARED_OBJECTS=		libtcc/libtcc.so libtcc/libtcc.so.$(VERSION)

DEPENDENCIES=		$(OBJECTS:%.o=%.d)

EXECUTABLES=		test/SimpleExample/SimpleExample\
			test/CorrelatorTest/CorrelatorTest
#			test/OpenCLCorrelatorTest/OpenCLCorrelatorTest

LIBRARIES=		-L$(CUDA)/lib64 \
			-L$(CUDA)/lib64/stubs -lcuda -lnvrtc #\
			#-L$(POWER_SENSOR)/lib -lpowersensor #-lnvidia-ml


%.d:			%.cc
			-$(CXX) $(CXXFLAGS) -MM -MT $@ -MT ${@:%.d=%.o} -MT ${@:%.d=%.s} $< -o $@

%.d:			%.cu
			-$(CXX) -x c++ $(CXXFLAGS) -MM -MT $@ -MT ${@:%.d=%.o} -MT ${@:%.d=%.s} $< -o $@

%.o:			%.cc
			$(CXX) $(CXXFLAGS) -o $@ -c $<

%.o:			%.cu
			$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.s:			%.cc
			$(CXX) $(CXXFLAGS) -o $@ -S $<

%.so:			%.so.$(VERSION)
			rm -f $@
			ln -s $(@F).$(VERSION) $@

all::			$(EXECUTABLES)

clean::
			$(RM) $(OBJECTS) $(SHARED_OBJECTS) $(DEPENDENCIES) $(EXECUTABLES)

libtcc/TCCorrelator.o:	libtcc/TCCorrelator.cu	# CUDA code embedded in object file
			ld -r -b binary -o $@ $<

libtcc/TCCorrelator.d:
			-

libtcc/libtcc.so.$(VERSION):		$(LIBTCC_OBJECTS)
			$(CXX) -shared -o $@ -Wl,-soname=$@ $^ $(LIBRARIES)

test/SimpleExample/SimpleExample:		$(SIMPLE_EXAMPLE_OBJECTS) libtcc/libtcc.so
			$(NVCC) $(NVCCFLAGS) -o $@ $(SIMPLE_EXAMPLE_OBJECTS) -Xlinker -rpath=. -Llibtcc -ltcc $(LIBRARIES)

test/CorrelatorTest/CorrelatorTest:	$(CORRELATOR_TEST_OBJECTS) libtcc/libtcc.so
			$(CXX) $(CXXFLAGS) -o $@ $(CORRELATOR_TEST_OBJECTS) -Wl,-rpath=. -Llibtcc -ltcc $(LIBRARIES)

test/OpenCLCorrelatorTest/OpenCLCorrelatorTest:	$(OPENCL_TEST_OBJECTS)
			$(CXX) $(CXXFLAGS) -o $@ $(OPENCL_TEST_OBJECTS) -lOpenCL

ifeq (0, $(words $(findstring $(MAKECMDGOALS), clean)))
-include $(DEPENDENCIES)
endif
