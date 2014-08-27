# This Makefile is used under Linux
# IN ORDER TO MAKE THINGS WORK, YOU NEED TO INSTALL BLAS (http://www.netlib.org/blas/) and the CWRAPPER CBLAS (http://www.netlib.org/blas/blast-forum/cblas.tgz):
# AFTER INSTALLING AND COMPILING, YOU NEED
# TO ADOPT THIS MAKEFILE TO FIT YOUR PATHS
HOME=/home/carpeng/Program/Matlab/KLR/kernel-logrec-multiclass
CBLASDIR = $(HOME)/CBLAS
BLASDIR =  $(HOME)/BLAS
MATLABDIR = /usr/local/matlab

# for Mac
# MATLABDIR ?= /opt/local/matlab

CXX = g++-4.4
#CXX = g++-4.2
CFLAGS = -fPIC -Wall -O3 -I$(MATLABDIR)/extern/include   -I../libklr -I$(CBLASDIR)/src

MEX = $(MATLABDIR)/bin/mex
MEX_OPTION = CC\#$(CXX) CXX\#$(CXX) CFLAGS\#"$(CFLAGS)" CXXFLAGS\#"$(CFLAGS)"
# comment the following line if you use MATLAB on 32-bit computer
MEX_OPTION += -largeArrayDims
MEX_EXT = $(shell $(MATLABDIR)/bin/mexext)

OCTAVEDIR ?= /usr/include/octave
OCTAVE_MEX = env CC=$(CXX) mkoctfile
OCTAVE_MEX_OPTION = --mex
OCTAVE_MEX_EXT = mex
OCTAVE_CFLAGS = -Wall -O3 -I$(OCTAVEDIR)

all:	matlab

matlab:	binary

octave:
	@make MEX="$(OCTAVE_MEX)" MEX_OPTION="$(OCTAVE_MEX_OPTION)" \
	MEX_EXT="$(OCTAVE_MEX_EXT)" CFLAGS="$(OCTAVE_CFLAGS)" \
	binary

binary: klr_train.$(MEX_EXT)

klr_train.$(MEX_EXT):     klr_train.cpp 
	$(MEX) $(MEX_OPTION) klr_train.cpp  ../libklr/libklr.cpp $(CBLASDIR)/lib/cblas_LINUX.a $(BLASDIR)/blas_LINUX.a -lgfortran

libklr:  
	cd .. && cd libklr && make

clean:	
	rm -f *~ *.o *.mex* *.obj
