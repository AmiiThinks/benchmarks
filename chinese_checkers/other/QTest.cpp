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
bool OLD_WAY = 1;
bool NEW_WAY = 0;

const int MAX_THREADS = 256;
const int memSize = 128;
int THREADS = 3;
bool done = false;

uint8_t *GetMemoryChunk(int whichThread);
void ReturnMemory(long whichThread, uint8_t *memory);
void *QueueWorker(void *queue);

LockQueue<uint8_t *> workQ[MAX_THREADS];
LockQueue<uint8_t *> memoryQ[MAX_THREADS];


int main(int argc, char **argv)
{
  Timer s;
  s.StartTimer();
  long entries = 1024*1024*512l;
  uint64_t *c = new uint64_t[entries];
  memset(c, 0, entries*sizeof(c[0]));
  printf("%1.2f spend building memory\n", s.EndTimer());

	if (argc > 1)
		THREADS = atoi(argv[1]);
	if (argc > 2)
	  {
	    if (strcmp(argv[2], "-old") == 0)
	      { OLD_WAY = 1; NEW_WAY = 0; }
	    if (strcmp(argv[2], "-new") == 0)
	      { OLD_WAY = 0; NEW_WAY = 1; }
	  }
	printf("Using %d threads\n", THREADS);
	if (OLD_WAY)
	  printf("Doing it the old way\n");
	if (NEW_WAY)
	  printf("Doing it the new way\n");
	Timer t;
	t.StartTimer();

	// dispatch # threads
	std::vector<pthread_t> theThreads(THREADS);
	for (int x = 0; x < THREADS; x++)
	{
		//pthread_t thread;
		pthread_create(&theThreads[x], NULL, QueueWorker, (void *)x);
		//theThreads.push_back(thread);
//		theThreads[x] = thread;
	}
	
	// Dispatch work
	for (uint64_t x = 0; x < WORK_UNITS; x++)
	{
		bool assigned = false;
		for (int y = 0; y < THREADS; y++)
		{
			if (NEW_WAY)
		    {
				uint8_t *mem = GetMemoryChunk(y);
				if (mem)
				{
					bzero(mem, memSize);
					workQ[y].Produce(mem);
					assigned = true;
					break;
				}
		    }
			if (OLD_WAY)
		    {
				if (workQ[y].NumEntries() < MAX_QUEUE_SIZE)
				{
					uint8_t *mem = GetMemoryChunk(y);
					bzero(mem, memSize);
					workQ[y].Produce(mem);
					assigned = true;
					break;
				}
		    }
		}
		if (!assigned)
		{
			usleep(100);
			x--;
		}
	}
	done = true;
	for (unsigned int x = 0; x < theThreads.size(); x++)
	{
		pthread_join(theThreads[x], 0);
	}
	printf("%1.4f seconds elapsed\n", t.EndTimer());
	getchar();
	return c[1843];
}

uint8_t *GetMemoryChunk(int whichThread)
{
  if (NEW_WAY)
  {
	  uint8_t *val;
	  if (memoryQ[whichThread].Empty())
		  return 0;
	  if (memoryQ[whichThread].Consume(val))
		  return val;
	  return 0;
  }
	
  if (OLD_WAY)
  {
	  for (int x = 0; x < THREADS; x++)
	  {
		  if (!memoryQ[x].Empty())
		  {
			  uint8_t *val;
			  if (memoryQ[x].Consume(val))
			  {
				  return val;
			  }
		  }
	  }
	  return new uint8_t[memSize];
  }
	
	return 0;
}

void ReturnMemory(long whichThread, uint8_t *memory)
{
	memoryQ[whichThread].Produce(memory);
}

void *QueueWorker(void *queue)
{
	long whichQueue = (long)queue;

	uint8_t memBuffer[memSize*MAX_QUEUE_SIZE];

	if (NEW_WAY)
	{
		for (int x = 0; x < MAX_QUEUE_SIZE; x++)
			memoryQ[whichQueue].Produce(&memBuffer[x*memSize]);
	}

	uint8_t *buffer;
	while (true)
	{
		if (workQ[whichQueue].Consume(buffer))
		{
			for (int x = 0; x < memSize; x++)
			{
				int sum = 0;
				for (int y = 0; y < x; y++)
					sum += y;
				buffer[x] = sum;
			}
			ReturnMemory(whichQueue, buffer);
		}
		else if (done)
		{
			break;
		}
		else {
			usleep(100);
		}
	}
	
	return (void*)0;
}
