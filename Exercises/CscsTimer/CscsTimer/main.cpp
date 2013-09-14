/**
 * @file		main.cpp
 * @brief   Cscs timer class usage example
 *
 * @author	Achille Peternier (achille.peternier@gmail.com), 2013
 */



//////////////
// #INCLUDE //
//////////////

   #include "cscs_timer.h"
#ifdef WIN32
   #include <Windows.h>
   #define sleepForSec(x) Sleep(x*1000)
#else
   #include <unistd.h>
   #define sleepForSec(x) sleep(x)
#endif
   #include <stdio.h>



//////////
// MAIN //
//////////

/**
 * App entry point.
 * @param argc number of args
 * @param argv arguments
 * @return 0 on success, error code otherwise
 */
int main(int argc, char *argv[])
{
   // Credits:
   printf("Simple timer class test - A. Peternier (C) 2013\n\n");
   
   // Init:
   CscsTimer::init();

   // Waste some time:
   for (int c=0; c<10; c++)
   {
      unsigned long long int startTime = CscsTimer::getCounter();
      unsigned long long int startRdtsc = CscsTimer::getRdtsc();
         sleepForSec(1);
      unsigned long long int endTime = CscsTimer::getCounter();
      unsigned long long int endRdtsc = CscsTimer::getRdtsc();      
      
      printf("   Time elapsed: %f ms, TSC: %llu cycles\n", CscsTimer::getCounterDiff(startTime, endTime), endRdtsc - startRdtsc);
   }

   // Done:
   return 0;
}
