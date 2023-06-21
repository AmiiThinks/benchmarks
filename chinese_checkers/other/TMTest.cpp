//
//  QTest.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 6/14/12.
//  Copyright (c) 2012 University of Denver. All rights reserved.
//

#include <iostream>
#include <stdint.h>
#include <strings.h>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include "Timer.h"
#include "LockQueue.h"
#include "LockFreeQueue.h"

#define WORK_UNITS 1000000
#define MAX_QUEUE_SIZE 50

#define WRITE_COUNT 1000000

bool OLD_WAY = 1;
bool NEW_WAY = 0;

const int MAX_THREADS = 256;
const int memSize = 128;
int THREADS = 3;
bool done = false;

uint8_t *GetMemoryChunk(int whichThread);
void ReturnMemory(long whichThread, uint8_t *memory);
void *QueueWorker(void *queue);
void *RandomQueueWorker(void *queue);

LockQueue<uint8_t *> workQ[MAX_THREADS];
LockQueue<uint8_t *> memoryQ[MAX_THREADS];

std::vector<bool> memory;


int main(int argc, char **argv)
{
  uint64_t entries = 62*8*1024*1024*1024ul;
  memory.resize(entries);
  bool random = false;

  if (argc > 1)
    {
      THREADS = atoi(argv[1]);
    }
  if (argc > 2)
    {
      if (strcmp(argv[2], "-random") == 0)
	random = true;
      else
	random = false;
    }
  printf("Using %d threads\n", THREADS);
  printf("%u entries allocated\n", entries);
  printf("%s random writes\n", random?"using":"not using");
  Timer t;
  t.StartTimer();
  
  // dispatch # threads
  std::vector<pthread_t> theThreads(THREADS);
  for (int x = 0; x < THREADS; x++)
	{
	  if (random)
	    {
	    pthread_create(&theThreads[x], NULL, RandomQueueWorker, (void *)x);
	    }
	  else {
		pthread_create(&theThreads[x], NULL, QueueWorker, (void *)x);
	  }
		//pthread_t thread;
		//theThreads.push_back(thread);
//		theThreads[x] = thread;
	}
	
	done = true;
	for (unsigned int x = 0; x < theThreads.size(); x++)
	{
		pthread_join(theThreads[x], 0);
	}
	t.EndTimer();
	printf("%1.4f seconds elapsed [%1.3f Mbits/second]\n", t.GetElapsedTime(), THREADS*WRITE_COUNT/(8.0*1e6*t.GetElapsedTime()));
	return 0;
}

void *RandomQueueWorker(void *queue)
{
  long whichQueue = (long)queue;
  uint64_t memSize = memory.size();
  for (int x = 0; x < WRITE_COUNT; x++)
    {
      uint64_t r1, r2;
      r1 = random();
      r2 = random();
      r1 = (r1<<32)|r2;
      memory[r1%memSize] = whichQueue;
    }
  return (void*)0;
}

void *QueueWorker(void *queue)
{
  long whichQueue = (long)queue;
  uint64_t memSize = memory.size();
  uint64_t block = memSize/THREADS;
  for (int x = 0; x < WRITE_COUNT; x++)
    {
      uint64_t r1, r2;
      r1 = random();
      r2 = random();
      r1 = (r1<<32)|r2;
      memory[block*whichQueue+(r1%block)] = whichQueue;
    }
  return (void*)0;
}
