#include <iostream>
#include "CCheckers.h"
#include <assert.h>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "ThreadedSolver.h"
#include "SimpleSolver.h"
#include "PrioritySolver.h"
#include "Timer.h"
#include <deque>

void TestSuccessors();
void FindMaxMinSuccessors(int threads);
int numThreads = 4;
uint64_t numStates = 100000000000ull;
void *Worker(void *division);
pthread_mutex_t lock;

int main(int argc, char * const argv[])
{
  if (argc > 1)
    numThreads = atoi(argv[1]);
  if (argc > 2)
    numStates = atoi(argv[2]);

  Timer t;
  t.StartTimer();
  std::vector<pthread_t> threads;
  threads.resize(numThreads);
  for (int x = 0; x < numThreads; x++)
    {
      pthread_create(&threads[x], NULL, Worker, (void *)x);
    }
  for (int x = 0; x < numThreads; x++)
    pthread_join(threads[x], 0);
  
  t.EndTimer();
  printf("%1.3f seconds elapsed\n", t.GetElapsedTime());
  return 0;
}

void *Worker(void *division)
{
  int t = numThreads;
  int states = numStates;
  uint64_t tot = 0;
  long cnt = (long)division;
  
  for (uint64_t x = cnt; x < numStates; x+=numThreads)
    {
      tot++;
    }
  
  pthread_exit(0);
}
