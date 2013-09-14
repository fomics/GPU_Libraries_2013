/**
 * @file		cscs_timer.h
 * @brief	A simple, portable class for measuring time
 *
 * @author	Achille Peternier (achille.peternier@gmail.com), 2013
 */
#ifndef CSCS_TIMER_H_INCLUDED
#define CSCS_TIMER_H_INCLUDED



/**
 * @brief Timing facilities.
 */
class CscsTimer
{
//////////
public: //
//////////   

   // Init (Win32 only):
   static bool init();   

   // Get/set:
   static unsigned long long int getRdtsc();
   static unsigned long long int getCounter();
   static double getCounterDiff(unsigned long long int t1, unsigned long long int t2);
   

///////////
private: //
///////////   

   // Vars:
   static unsigned long long int cpuFreq;   ///< Frequency multiplier used by performance counter (on Win32)   
};


///////////////////////////////
#endif // CSCS_TIMER_H_INCLUDED
