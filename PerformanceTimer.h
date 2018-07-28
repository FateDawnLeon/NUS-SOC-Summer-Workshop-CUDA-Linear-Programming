// PerformanceTimer.h - High Resolution Timer
// Since not every PC has performance hardware, we have fallback mechanism
// to timeGetTime()
//

#ifndef __PERFORMANCETIMER_H__
#define __PERFORMANCETIMER_H__

double get_wall_time();
double get_cpu_time();

#endif