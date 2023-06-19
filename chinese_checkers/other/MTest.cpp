#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "Timer.h"

typedef uint8_t mySize_t;

unsigned long entries = 512*1024*1024l*8/sizeof(mySize_t);


int main(int argc, char **argv)
{
  Timer t, u;
  mySize_t *data;
  data = new mySize_t[entries];
  t.StartTimer();
  for (unsigned long x = 0; x < entries; x++)
    data[x] = 0;
  printf("%1.2f elapsed in write\n", t.EndTimer());
  printf("%1.2fGBit/s write bandwidth\n", entries*sizeof(data[0])/(t.GetElapsedTime()*1e9));
  u.StartTimer();
  long sum = 0;
  for (unsigned long x = 0; x < entries; x++)
    sum += data[x];
  printf("%1.2f elapsed in read\n", u.EndTimer());
  printf("%1.2fGBit/s write bandwidth\n", entries*sizeof(data[0])/(u.GetElapsedTime()*1e9));


  t.StartTimer();
  memset(data, 0, entries*sizeof(data[0]));
printf("%1.2f elapsed in memset\n", t.EndTimer());
printf("%1.2fGBit/s write bandwidth\n", entries*sizeof(data[0])/(t.GetElapsedTime()*1e9));
  return sum;
}
