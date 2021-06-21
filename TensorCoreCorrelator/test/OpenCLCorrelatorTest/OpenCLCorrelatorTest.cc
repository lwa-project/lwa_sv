#define NR_BITS 8
#define NR_CHANNELS 480
#define NR_POLARIZATIONS 2
#define NR_SAMPLES_PER_CHANNEL 3072
#define NR_RECEIVERS 576
#define NR_BASELINES ((NR_RECEIVERS) * ((NR_RECEIVERS) + 1) / 2)
#define NR_RECEIVERS_PER_BLOCK 64
#define NR_TIMES_PER_BLOCK (128 / (NR_BITS))


#include "test/Common/ComplexInt4.h"

#include <complex>
#include <exception>
//#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>
#include <sys/types.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <CL/cl_ext.h>


#if NR_BITS == 4
typedef complex_int4_t        Sample;
typedef std::complex<int32_t> Visibility;
#elif NR_BITS == 8
typedef std::complex<int8_t>  Sample;
typedef std::complex<int32_t> Visibility;
#elif NR_BITS == 16
typedef std::complex<__half>  Sample;
typedef std::complex<float>   Visibility;
#endif

typedef Sample Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_RECEIVERS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
typedef Visibility Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];


std::string errorMessage(cl_int error)
{
  switch (error) {
  case CL_SUCCESS:                            return "Success!";
  case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
  case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
  case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
  case CL_OUT_OF_RESOURCES:                   return "Out of resources";
  case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
  case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
  case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
  case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
  case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
  case CL_MAP_FAILURE:                        return "Map failure";
  case CL_INVALID_VALUE:                      return "Invalid value";
  case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
  case CL_INVALID_PLATFORM:                   return "Invalid platform";
  case CL_INVALID_DEVICE:                     return "Invalid device";
  case CL_INVALID_CONTEXT:                    return "Invalid context";
  case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
  case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
  case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
  case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
  case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
  case CL_INVALID_SAMPLER:                    return "Invalid sampler";
  case CL_INVALID_BINARY:                     return "Invalid binary";
  case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
  case CL_INVALID_PROGRAM:                    return "Invalid program";
  case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
  case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
  case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
  case CL_INVALID_KERNEL:                     return "Invalid kernel";
  case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
  case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
  case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
  case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
  case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
  case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
  case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
  case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
  case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
  case CL_INVALID_EVENT:                      return "Invalid event";
  case CL_INVALID_OPERATION:                  return "Invalid operation";
  case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
  case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
  case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
  case CL_INVALID_GLOBAL_WORK_SIZE:           return "Invalid global work size";
#if defined CL_INVALID_PROPERTY
  case CL_INVALID_PROPERTY:                   return "Invalid property";
#endif
#if defined CL_INVALID_IMAGE_DESCRIPTOR
  case CL_INVALID_IMAGE_DESCRIPTOR:           return "Invalid image descriptor";
#endif
#if defined CL_INVALID_COMPILER_OPTIONS
  case CL_INVALID_COMPILER_OPTIONS:           return "Invalid compiler options";
#endif
#if defined CL_INVALID_LINKER_OPTIONS
  case CL_INVALID_LINKER_OPTIONS:             return "Invalid linker options";
#endif
#if defined CL_INVALID_DEVICE_PARTITION_COUNT
  case CL_INVALID_DEVICE_PARTITION_COUNT:     return "Invalid device partition count";
#endif
  default:                                    std::stringstream str;
    str << "Unknown (" << error << ')';
    return str.str();
  }
}


void createContext(cl::Context &context, std::vector<cl::Device> &devices)
{
  const char *platformName = getenv("PLATFORM");

#if defined __linux__
  if (platformName == 0)
#endif
  //platformName = "Intel(R) FPGA SDK for OpenCL(TM)";
  //platformName = Intel(R) OpenCL";
  platformName = "NVIDIA CUDA";
  //platformName = "AMD Accelerated Parallel Processing";

  cl_device_type type = CL_DEVICE_TYPE_DEFAULT;

  const char *deviceType = getenv("TYPE");

  if (deviceType != 0) {
    if (strcmp(deviceType, "GPU") == 0)
      type = CL_DEVICE_TYPE_GPU;
    else if (strcmp(deviceType, "CPU") == 0)
      type = CL_DEVICE_TYPE_CPU;
    else if (strcmp(deviceType, "ACCELERATOR") == 0)
      type = CL_DEVICE_TYPE_ACCELERATOR;
    else
      std::cerr << "Unrecognized device type: " << deviceType;
  }

  const char *deviceName = getenv("DEVICE");

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  for (cl::Platform &platform : platforms) {
    std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::clog << "Platform version: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
    std::clog << "Platform extensions: " << platform.getInfo<CL_PLATFORM_EXTENSIONS>() << std::endl;
  }

  for (cl::Platform &platform : platforms) {
    if (strcmp(platform.getInfo<CL_PLATFORM_NAME>().c_str(), platformName) == 0) {
      platform.getDevices(type, &devices);

      for (cl::Device &device : devices) {
	std::clog << "Device: " << device.getInfo<CL_DEVICE_NAME>() << ", "
		      "capability: " << device.getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV>() << '.' << device.getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV>()
//#if defined CL_DEVICE_TOPOLOGY_AMD
//		      ", PCIe bus 0x" << std::hex << std::setiosflags (std::ios::uppercase) << std::setfill('0') << std::setw(2) << (device.getInfo<CL_DEVICE_TOPOLOGY_AMD>().pcie.bus & 255) << std::dec
//#endif
	          << std::endl;
      }

      context = cl::Context(devices);
      return;
    }
  }

  std::cerr << "Platform not found: \"" << platformName << '"' << std::endl;
  exit(1);
}


cl::Program createProgramFromSources(cl::Context &context, std::vector<cl::Device> &devices, const std::string &sources, const char *args)
{
  std::stringstream cmd;
  cmd << "#include \"" << sources << '"' << std::endl;
#if defined CL_VERSION_1_2
  cl::Program program(context, cmd.str());
#else
  std::string str = cmd.str();
  cl::Program::Sources src(1, std::make_pair(str.c_str(), str.length()));
  cl::Program program(context, src);
#endif

  try {
    program.build(devices, args);
    std::string msg;
    program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &msg);

    std::clog << msg << std::endl;
  } catch (cl::Error &error) {
    if (strcmp(error.what(), "clBuildProgram") == 0) {
      std::string msg;
      program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &msg);

      std::cerr << msg << std::endl;
      exit(1);
    } else {
      throw;
    }
  }

#if 0
  std::vector<size_t> binarySizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
  std::vector<char *> binaries = program.getInfo<CL_PROGRAM_BINARIES>();

  for (unsigned i = 0; i < binaries.size(); i++) {
    std::stringstream filename;
    filename << sources << '-' << i << ".ptx";
    std::ofstream(filename.str().c_str(), std::ofstream::binary).write(binaries[i], binarySizes[i]);
  }

  for (unsigned b = 0; b < binaries.size(); b++)
    delete [] binaries[b];
#endif

  return program;
}


cl::Program createProgramFromBinaries(cl::Context &context, std::vector<cl::Device> &devices)
{
#if 1
  cl::Program::Binaries binaries(devices.size());
  std::vector<std::string> ptx_code(devices.size());

  for (unsigned device = 0; device < devices.size(); device ++) {
    unsigned compute_capability = 10 * devices[device].getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV>() + devices[device].getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV>();
    std::stringstream command;

    command << "nvcc"
	    << " -ptx"
	    << " -arch=compute_" << compute_capability
	    << " -code=sm_" << compute_capability
	    << " -DNR_BITS=" << NR_BITS
	    << " -DNR_RECEIVERS=" << NR_RECEIVERS
	    << " -DNR_CHANNELS=" << NR_CHANNELS
	    << " -DNR_SAMPLES_PER_CHANNEL=" << NR_SAMPLES_PER_CHANNEL
	    << " -DNR_POLARIZATIONS=" << NR_POLARIZATIONS
	    << " -DNR_RECEIVERS_PER_BLOCK=" << NR_RECEIVERS_PER_BLOCK
	    << " -o -"
	    << " libtcc/TCCorrelator.cu"
	    << "|sed -e s/.param\\ .[a-zA-Z0-9]*/\\&\\ .ptr\\ .global/";

    std::clog << "executing: " << command.str() << std::endl;
    FILE *pipe = popen(command.str().c_str(), "r");

    while (!feof(pipe)) {
      char buffer[1024];

      if (fgets(buffer, 1024, pipe) != NULL)
       ptx_code[device] += buffer;
    }

    binaries[device] = (std::make_pair(ptx_code[device].c_str(), ptx_code[device].length()));
    pclose(pipe);
  }

#else
  std::ifstream ifs(name, std::ios::in | std::ios::binary);
  std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

  cl::Program::Binaries binaries(devices.size(), std::make_pair(str.c_str(), str.length()));
#endif

  cl::Program program(context, devices, binaries);
  program.build();

  return program;
} 


void setTestPattern(cl::CommandQueue &queue, cl::Buffer &samplesBuffer)
{
  Samples &samples = * (Samples *) queue.enqueueMapBuffer(samplesBuffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(Samples));

#if 0
  memset(samples, 0, sizeof(Samples));

  unsigned channel = NR_CHANNELS / 3;
  unsigned time    = NR_SAMPLES_PER_CHANNEL / 5;
  unsigned recv0   = NR_RECEIVERS > 174 ? 174 : NR_RECEIVERS / 3;
  unsigned recv1   = NR_RECEIVERS > 418 ? 418 : NR_RECEIVERS / 2;

#if NR_BITS == 4
  samples[channel][time / NR_TIMES_PER_BLOCK][recv0][POL_X][time % NR_TIMES_PER_BLOCK] = complex_int4_t(2, 3);
  samples[channel][time / NR_TIMES_PER_BLOCK][recv1][POL_X][time % NR_TIMES_PER_BLOCK] = complex_int4_t(4, 5);
#elif NR_BITS == 8
  samples[channel][time / NR_TIMES_PER_BLOCK][recv0][POL_X][time % NR_TIMES_PER_BLOCK] = std::complex<int8_t>(2, 3);
  samples[channel][time / NR_TIMES_PER_BLOCK][recv1][POL_X][time % NR_TIMES_PER_BLOCK] = std::complex<int8_t>(4, 5);
#elif NR_BITS == 16
  samples[channel][time / NR_TIMES_PER_BLOCK][recv0][POL_X][time % NR_TIMES_PER_BLOCK] = std::complex<__half>(2.0f, 3.0f);
  samples[channel][time / NR_TIMES_PER_BLOCK][recv1][POL_X][time % NR_TIMES_PER_BLOCK] = std::complex<__half>(4.0f, 5.0f);
#endif
#else
  Sample randomValues[7777];

  for (unsigned i = 0; i < 7777; i ++)
#if NR_BITS == 4
    randomValues[i] = complex_int4_t(16 * drand48() - 8, 16 * drand48() - 8);
#elif NR_BITS == 8
    randomValues[i] = std::complex<int8_t>(256 * drand48() - 128, 256 * drand48() - 128);
#elif NR_BITS == 16
    randomValues[i] = std::complex<__half>(drand48() - .5, drand48() - .5);
#endif

#pragma omp parallel for schedule (dynamic)
  for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
    for (unsigned time_major = 0, i = channel; time_major < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK; time_major ++)
      for (unsigned receiver = 0; receiver < NR_RECEIVERS; receiver ++)
        for (unsigned polarization = 0; polarization < NR_POLARIZATIONS; polarization ++)
          for (unsigned time_minor = 0; time_minor < NR_TIMES_PER_BLOCK; time_minor ++, i ++)
            samples[channel][time_major][receiver][polarization][time_minor] = randomValues[i % 7777U];
#endif

  queue.enqueueUnmapMemObject(samplesBuffer, samples);
}


void checkTestPattern(cl::CommandQueue &queue, cl::Buffer &visibilitiesBuffer, cl::Buffer &samplesBuffer)
{
  Visibilities &visibilities = * (Visibilities *) queue.enqueueMapBuffer(visibilitiesBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(Visibilities));

  std::atomic<int> count(0);

#if 0
  Samples &samples = * (Samples *) queue.enqueueMapBuffer(samplesBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(Samples));

  std::cout << "checking ..." << std::endl;
    
//#pragma omp parallel for schedule (dynamic)
  for (unsigned channel = 0; channel < NR_CHANNELS; channel ++) {
//#pragma omp critical (cout)
    //std::cout << "channel = " << channel << std::endl;

#if NR_BITS == 4 || NR_BITS == 8
    typedef std::complex<int32_t> Sum[NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];
#elif NR_BITS == 16
    typedef std::complex<float> Sum[NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];
#endif

    Sum &sum = *new Sum[1];
    memset(sum, 0, sizeof sum);

    for (unsigned major_time = 0; major_time < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK; major_time ++)
      for (unsigned recv1 = 0, baseline = 0; recv1 < NR_RECEIVERS; recv1 ++)
        for (unsigned recv0 = 0; recv0 <= recv1; recv0 ++, baseline ++)
          for (unsigned minor_time = 0; minor_time < NR_TIMES_PER_BLOCK; minor_time ++)
            for (unsigned pol0 = 0; pol0 < NR_POLARIZATIONS; pol0 ++)
              for (unsigned pol1 = 0; pol1 < NR_POLARIZATIONS; pol1 ++)
#if NR_BITS == 4 || NR_BITS == 8
                sum[baseline][pol1][pol0] += (std::complex<int32_t>) samples[channel][major_time][recv1][pol1][minor_time] * conj((std::complex<int32_t>) samples[channel][major_time][recv0][pol0][minor_time]);
#elif NR_BITS == 16
                sum[baseline][pol1][pol0] += samples[channel][major_time][recv1][pol1][minor_time] * conj(samples[channel][major_time][recv0][pol0][minor_time]);
#endif

    for (unsigned baseline = 0; baseline < NR_BASELINES; baseline ++)
      for (unsigned pol0 = 0; pol0 < NR_POLARIZATIONS; pol0 ++)
        for (unsigned pol1 = 0; pol1 < NR_POLARIZATIONS; pol1 ++)
#if NR_BITS == 4 || NR_BITS == 8
          if (visibilities[channel][baseline][pol1][pol0] != sum[baseline][pol1][pol0] && ++ count < 10)
#pragma omp critical (cout)
            std::cout << "visibilities[" << channel << "][" << baseline << "][" << pol1 << "][" << pol0 << "], expected " << sum[baseline][pol1][pol0] << ", got " << visibilities[channel][baseline][pol1][pol0] << std::endl;
#elif NR_BITS == 16
          float absolute = abs(visibilities[channel][baseline][pol1][pol0] - sum[baseline][pol1][pol0]);
          float relative = abs(visibilities[channel][baseline][pol1][pol0] / sum[baseline][pol1][pol0]);
                                                                        195,1         83%


          if ((relative < .999 || relative > 1.001) && absolute > .0001 * NR_SAMPLES_PER_CHANNEL && ++ count < 10)
              std::cout << "visibilities[" << channel << "][" << baseline << "][" << pol1 << "][" << pol0 << "], expected " << sum[baseline][pol1][pol0] << ", got " << visibilities[channel][baseline][pol1][pol0] << ", relative = " << relative << ", absolute = " << absolute << std::endl;
#endif
    delete &sum;
  }

  std::cout << "#errors = " << count << std::endl;

  queue.enqueueUnmapMemObject(samplesBuffer, samples);
#else
  for (unsigned channel = 0; channel < NR_CHANNELS; channel ++)
    for (unsigned baseline = 0; baseline < NR_BASELINES; baseline ++)
      for (unsigned pol0 = 0; pol0 < NR_POLARIZATIONS; pol0 ++)
        for (unsigned pol1 = 0; pol1 < NR_POLARIZATIONS; pol1 ++)
#if NR_BITS == 4 || NR_BITS == 8
            if (visibilities[channel][baseline][pol0][pol1] != std::complex<int32_t>(0, 0))
#elif NR_BITS == 16
            if (visibilities[channel][baseline][pol0][pol1] != std::complex<float>(0.0f, 0.0f))
#endif
              if (count ++ < 10)
#pragma omp critical (cout)
                std::cout << "visibilities[" << channel << "][" << baseline << "][" << pol0 << "][" << pol1 << "] = " << visibilities[channel][baseline][pol0][pol1] << std::endl;
#endif

  queue.enqueueUnmapMemObject(visibilitiesBuffer, visibilities);
}


int main()
{
  try {
    cl::Context             context;
    std::vector<cl::Device> devices;
    createContext(context, devices);

    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    cl::Buffer samples(context, CL_MEM_READ_ONLY /*| CL_MEM_HOST_WRITE_ONLY*/, sizeof(Samples));
    cl::Buffer visibilities(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(Visibilities));

    setTestPattern(queue, samples);

    cl::Program program(createProgramFromBinaries(context, devices));

    cl::Kernel correlatorKernel(program, "correlate");
    correlatorKernel.setArg(0, visibilities);
    correlatorKernel.setArg(1, samples);

    cl::Event event;

#if NR_RECEIVERS_PER_BLOCK == 32 || NR_RECEIVERS_PER_BLOCK == 48
    unsigned nrBlocks = ((NR_RECEIVERS + NR_RECEIVERS_PER_BLOCK - 1) / NR_RECEIVERS_PER_BLOCK) * ((NR_RECEIVERS + NR_RECEIVERS_PER_BLOCK - 1) / NR_RECEIVERS_PER_BLOCK + 1) / 2;
#elif NR_RECEIVERS_PER_BLOCK == 64
    unsigned nrBlocks = ((NR_RECEIVERS + NR_RECEIVERS_PER_BLOCK - 1) / NR_RECEIVERS_PER_BLOCK) * ((NR_RECEIVERS + NR_RECEIVERS_PER_BLOCK - 1) / NR_RECEIVERS_PER_BLOCK);
#endif

    cl::NDRange globalWorkSize(32 * nrBlocks, 2 * NR_CHANNELS, 2 * 1);
    cl::NDRange localWorkSize(32, 2, 2);

for (int i = 0; i < 100; i ++)
    queue.enqueueNDRangeKernel(correlatorKernel, cl::NullRange, globalWorkSize, localWorkSize, 0, &event);
    queue.finish();

    double runtime = (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
    unsigned long long nrOperations = 8LLU * (NR_RECEIVERS * NR_RECEIVERS + 1) / 2 * NR_POLARIZATIONS * NR_POLARIZATIONS * NR_CHANNELS * NR_SAMPLES_PER_CHANNEL;
    double TFLOPS = nrOperations / runtime * 1e-12;
    std::clog << "runtime = " << runtime << "s, TFLOPS = " << TFLOPS << std::endl;

    checkTestPattern(queue, visibilities, samples);
   } catch (cl::Error &error) {
     std::cerr << "caught cl::Error: " << error.what() << ": " << errorMessage(error.err()) << std::endl;
   } catch (std::exception &error) {
     std::cerr << "caught std::exception: " << error.what() << std::endl;
   }

  return 0;
}
