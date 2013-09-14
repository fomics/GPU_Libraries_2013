/**
 * @file    hwinfo.cpp
 * @brief   Gathers information on underlying HW and OpenCL
 *
 * @author  Achille Peternier (achille.peternier@gmail.com), 2013
 */



//////////////
// #INCLUDE //
//////////////

   // #include <hwloc.h>
   #include <iostream>   
   #include <CL/cl.h>
   using namespace std;



/////////////
// #DEFINE //
/////////////

   // General:
   #define APP_NAME "HwInfo"
   #define nullptr NULL ///< ugly c++0x workaround


///////////////////
// LOCAL METHODS //
///////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Scan the underlying system for its hw specs.
 * @return true on success, false on failure
 */
/*bool scanHardware()
{
   hwloc_topology_t topology;
   hwloc_obj_t obj;

   hwloc_topology_init(&topology);
   hwloc_topology_load(topology);

   long long ram = 0;
   int numberOfSystems = 0;
   int numberOfNumaNodes = 0;
   int numberOfCpus = 0;
   int numberOfCores = 0;
   int numberOfPus = 0;

   // Get system memory:
   int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_MACHINE);
   if (depth != HWLOC_TYPE_DEPTH_UNKNOWN)
      numberOfSystems = hwloc_get_nbobjs_by_depth(topology, depth);
   hwloc_obj_t memo = hwloc_get_obj_by_depth(topology, depth, 0);
   ram = memo->memory.total_memory;

   // Get NUMA nodes:
   depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NODE);
   if (depth != HWLOC_TYPE_DEPTH_UNKNOWN)
      numberOfNumaNodes = hwloc_get_nbobjs_by_depth(topology, depth);

   // Get CPUs:
   depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NODE);
   if (depth != HWLOC_TYPE_DEPTH_UNKNOWN)
      numberOfCpus = hwloc_get_nbobjs_by_depth(topology, depth);

   // Get cores:
   depth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
   if (depth != HWLOC_TYPE_DEPTH_UNKNOWN)
      numberOfCores = hwloc_get_nbobjs_by_depth(topology, depth);

   // Get PUs:
   depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
   if (depth != HWLOC_TYPE_DEPTH_UNKNOWN)
      numberOfPus = hwloc_get_nbobjs_by_depth(topology, depth);

   // Report:
   cout << " Available RAM . . . :  " << (ram / 1024 / 1024) << " MB " << endl;
   cout << " Nr. of systems  . . :  " << numberOfSystems << endl;
   cout << " Nr. of NUMA nodes . :  " << numberOfNumaNodes << endl;
   cout << " Nr. of CPUs . . . . :  " << numberOfCpus << endl;
   cout << " Nr. of cores  . . . :  " << numberOfCores << endl;
   cout << " Nr. of PUs  . . . . :  " << numberOfPus << endl;

   // Release:
   hwloc_topology_destroy(topology);

   // Done:   
   return true;
}*/


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Print per-opencl-device internal list of properties.
 * @param countId sequential counter for devices
 * @param id device id
 * @param defaultId default device id (to highlight the default one)
 * @return true on success, false on failure
 */
bool reportOclDeviceProps(int countId, cl_device_id id, cl_device_id defaultId)
{
   // Name:
   char deviceName[1024];
   clGetDeviceInfo(id, CL_DEVICE_NAME, 1024, deviceName, nullptr);

   // Trim text:
   char *str = deviceName;
   char *end;
   while (isspace(*str)) str++;
   if (*str == 0)
      return false;

   if (id != defaultId)
      cout << "    Device " << countId << "  . . :  " << str << endl;
   else
      cout << "    Device " << countId << "  . . :  " << str << " (default)" << endl;

   // Version:
   char deviceVersion[1024];
   clGetDeviceInfo(id, CL_DEVICE_VERSION, 1024, deviceVersion, nullptr);
   cout << "    Device version:  " << deviceVersion << endl;

   // Driver:
   char driverVersion[1024];
   clGetDeviceInfo(id, CL_DRIVER_VERSION, 1024, driverVersion, nullptr);
   cout << "    Driver version:  " << driverVersion << endl;

   // Global memory:
   cl_ulong globalMemorySize = 0;
   clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemorySize), &globalMemorySize, nullptr);
   cout << "    Glob. mem size:  " << (globalMemorySize / 1024 / 1024) << " MB" << endl;

   // Local memory:
   cl_ulong localMemorySize = 0;
   clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemorySize), &localMemorySize, nullptr);
   cout << "    Loc. mem size :  " << (localMemorySize / 1024) << " KB" << endl;

   // Max mem object:
   cl_ulong maxMemObject = 0;
   clGetDeviceInfo(id,  CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxMemObject), &maxMemObject, nullptr);
   cout << "    Max. mem obj. :  " << (maxMemObject / 1024 / 1024) << " MB" << endl;  

   // Compute units:
   cl_uint computeUnits = 0;
   clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
   cout << "    Compute units :  " << computeUnits << endl;

   // Global cache:
   cl_ulong globalCache = 0;
   clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(globalCache), &globalCache, nullptr);
   cout << "    Global cache  :  " << (globalCache / 1024) << " KB" << endl;

   // Processor clock:
   cl_uint maxClock = 0;
   clGetDeviceInfo(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClock), &maxClock, nullptr);
   cout << "    Max clock . . :  " << maxClock << " MHz" << endl;

   // Max workgroup size:
   size_t maxWorkgroupSize = 0;
   clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkgroupSize), &maxWorkgroupSize, nullptr);
   cout << "    Max workgroup :  " << maxWorkgroupSize << endl;

   // Done:
   return true;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Scan the underlying system for OpenCL platforms and their specs.
 * @return true on success, false on failure
 */
bool scanOpenClPlatforms()
{
   cl_uint numberOfPlatforms;
   cl_platform_id* clPlatformIds;
   cl_int ciErrNum;

   // Get OpenCL platform count:
   ciErrNum = clGetPlatformIDs(0, nullptr, &numberOfPlatforms);
   if (ciErrNum != CL_SUCCESS)
   {
      cout << "ERROR: unable to enumerate OpenCL platforms" << endl;
      return false;
   }

   // No platforms?
   if (numberOfPlatforms == 0)
   {
      cout << "No OpenCL platform found" << endl;
      return false;
   }

   // Report:
   cout << " Platforms found  :  " << numberOfPlatforms << endl;

   // Make space for IDs:
   clPlatformIds = reinterpret_cast<cl_platform_id *>(new char[numberOfPlatforms * sizeof(cl_platform_id)]);
   if (clPlatformIds == nullptr)
   {
      cout << "ERROR: failed to allocate memory for platform ID(s)" << endl;
      return false;
   }

   // Get platform info for each platform and trap the NVIDIA platform if found:
   ciErrNum = clGetPlatformIDs (numberOfPlatforms, clPlatformIds, nullptr);
   for (cl_uint c = 0; c < numberOfPlatforms; c++)
   {
      cl_uint numberOfGpus = 0;
      cl_uint numberOfCpus = 0;
      cl_uint defaultDevice = 0;
      cl_ulong globalMemorySize = 0;
      const int BUFFER_SIZE = 1024;
      char buffer[BUFFER_SIZE];
      ciErrNum = clGetPlatformInfo(clPlatformIds[c], CL_PLATFORM_NAME, BUFFER_SIZE, &buffer, nullptr);
      if (ciErrNum == CL_SUCCESS)
      {
         // Report:
         cout << "  Platform " << c << "  . . :  " << buffer << endl;

         cl_device_id devId[BUFFER_SIZE];
         cl_device_id defaultId;

         // Default:
         clGetDeviceIDs(clPlatformIds[c], CL_DEVICE_TYPE_DEFAULT, 1, &defaultId, &defaultDevice);

         // Show GPU-based devices:
         clGetDeviceIDs(clPlatformIds[c], CL_DEVICE_TYPE_GPU, BUFFER_SIZE, devId, &numberOfGpus);
         cout << "   GPUs . . . . . :  " << numberOfGpus << endl;
         for (int d = 0; d < numberOfGpus; d++)
            reportOclDeviceProps(d, devId[d], defaultId);

         // ...CPU-based devices:
         clGetDeviceIDs(clPlatformIds[c], CL_DEVICE_TYPE_CPU, BUFFER_SIZE, devId, &numberOfCpus);
         cout << "   CPUs . . . . . :  " << numberOfCpus << endl;
         for (int d = 0; d < numberOfCpus; d++)
            reportOclDeviceProps(d, devId[d], defaultId);
         cout << endl;
      }
   }
   delete [] clPlatformIds;

   // Done:
   return true;
}



//////////
// MAIN //
//////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Main method.
 * @param argc argument count
 * @param argv argument values
 * @return 0 on success, error code otherwise
 */
int main(int argc, char *argv[])
{
   // Credits:
   cout << APP_NAME << ", A. Peternier (C) 2013" << endl << endl << endl;


   ////////////
   // Parse HW:
   /*cout << "------------------" << endl;
   cout << "Hardware topology:" << endl;
   cout << "------------------" << endl << endl;
   scanHardware();
   cout << endl << endl;*/


   ////////////////////
   // OpenCL platforms:
   cout << "-----------------" << endl;
   cout << "OpenCL platforms:" << endl;
   cout << "-----------------" << endl << endl;
   scanOpenClPlatforms();

   // Done:
   return 0;
}
