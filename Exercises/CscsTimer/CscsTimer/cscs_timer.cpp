/**
 * @file		cscs_timer.cpp
 * @brief   A simple, portable class for measuring time
 *
 * @author	Achille Peternier (achille.peternier@gmail.com), 2013
 */



//////////////
// #INCLUDE //
//////////////   

   #include "cscs_timer.h"   
#ifdef WIN32
   #include <Windows.h>
   #include <intrin.h>
   #pragma intrinsic(__rdtsc)
#else // Linux
   #include <time.h>
   #include <sys/time.h>
#endif
   #include <stdio.h>



////////////
// STATIC //
////////////

   // Vars:
   unsigned long long int CscsTimer::cpuFreq = 0;



/////////////////////////////
// BODY OF CLASS CscsTimer //
/////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Init static components of the class.
 */
bool CscsTimer::init()
{
#ifdef WIN32   
   if (!QueryPerformanceFrequency((LARGE_INTEGER *) &cpuFreq))
   {
      printf("ERROR: Timer performance counter not supported\n");      
      return false;
   }
#else // Linux
#endif

   // Done:
   return true;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Get current RDTSC timestamp.
 * @return RDTSC timestamp
 */
unsigned long long int CscsTimer::getRdtsc()
{
#ifdef WIN32
   return __rdtsc();
#else // Linux
   unsigned long long int val;
   unsigned int __a,__d;
   asm volatile("rdtsc" : "=a" (__a), "=d" (__d));
   val = ((unsigned long)__a) | (((unsigned long)__d)<<32);
   return val;
#endif
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Get current time elapsed in ticks.
 * @return Time elapsed in ticks
 */
unsigned long long int CscsTimer::getCounter()
{
#ifdef WIN32
   // Not supported?
   if (cpuFreq == 0)
      return 0;

   unsigned long long int li;
   QueryPerformanceCounter((LARGE_INTEGER *) &li);
   return li;
#else // Linux
   timeval t;
   gettimeofday(&t, NULL);
   unsigned long long int ticks = t.tv_sec * 1000000 + t.tv_usec;
   return ticks;
#endif
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Get time interval in milliseconds between two tick snapshots.
 * @param t1 start time
 * @param t2 end time
 * @return Time elapsed in milliseconds
 */
double CscsTimer::getCounterDiff(unsigned long long int t1, unsigned long long int t2)
{
#ifdef WIN32
   // Not supported?
   if (cpuFreq == 0)
      return 0.0;

   unsigned long long int r = ((t2 - t1) * 1000000) / cpuFreq;
   return (double) r / 1000.0;
#else // Linux
   unsigned long long int r = t2 - t1;
   return (double) r / 1000.0;
#endif
}
