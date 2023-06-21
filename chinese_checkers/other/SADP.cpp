//
//  SADP.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 10/26/11.
//  Copyright (c) 2011 University of Denver. All rights reserved.
//

#include <iostream>
#include <unistd.h>
#include "LockFreeQueue.h"
#include "LockQueue.h"
#include "CCheckers.h"

#define DATA_SIZE 4096
#define NUM_THREADS 2

struct workerData;

typedef LockQueue<workerData> Q;
typedef LockQueue<uint64_t> WriteQ;

struct writerData {
	WriteQ q[NUM_THREADS];
};

struct workerData {
	uint8_t *data;
	uint8_t currentTag;
	uint32_t entries;
	uint64_t entriesOffset;
	uint64_t rangeStart;
	uint64_t rangeEnd;
	WriteQ *wq;
};

int currentDistance = 0;

std::vector<uint8_t> db;

int main(int argc, char **argv)
{
	CCheckers cc;
	CCState s;
	
	cc.Reset(s);
	
	
	return 0;
}

// the data writer writes any updates back to memory buffer
void *DataWriter(void *data)
{
	writerData *data = (writerData*)data;
	while (1)
	{
		for (int x = 0; x < NUM_THREADS; x++)
		{
			uint64_t val;
			if (data->q[x]->Consume(val))
			{
				db[val] = currentDistance;
			}
		}
		usleep(10); // 10 microseconds
	}
}

int ReadDBValue(uint64_t index, uint8_t *data)
{
#ifdef BITS2
	int val = data[index>>2]>>(2*(index%4));
	return val&0x3;
#else
	return data[index];
#endif
}

// the working gets work chunks from the queue and finds nodes at the next cost boundary
void *IncrementalWorker(void *queue)
{
	Q *q = (Q*)queue;
	CCheckers cc;
	CCState s;
	while (1)
	{
		workerData d;
		if (q->Consume(d))
		{
			for (int x = 0; x < d.entries; x++)
			{
				if (ReadDBValue(x, d.data) == d.currentTag)
				{
					cc.unrankPlayer(d.entriesOffset+x, s, 0);
					CCMove *m = cc.getMoves(s);
					for (CCMove *t = m; t; t = t->next)
					{
						cc.ApplyMove(s, t);
						uint64_t rank = cc.rankPlayer(s, 0);
						if (rank >= d.rangeStart && rank < d.rangeEnd)
						{
							// put on write queue
							d.wq->Produce(rank);
						}
						cc.UndoMove(s, t);
					}
					cc.freeMove(m);
				}
			}
		}
		else {
			usleep(10); // 10 microseconds
		}
	}
	pthread_exit(NULL);
}

/*
 
 7.44GB (max) of RAM to store bit map for changed states
 3.9GB of RAM to store rough open list [512(!) states per bit]
 ------
 11.3GB
 
 Repeat ~40 times
 
 Repeat 20 times
 
 * Iterate through all data and find states changed in the last iteration.
 * Update children in current memory
 
 -- 16 thread helpers:
 --- give them a chunk of memory & ask to find successors
 --- successors passed to writing thread
 
 -- reading thread
 --- reads memory from disk
 --- passes to thread helpers
 
 -- writing thread
 --- reads from queues & writes to large buffer
 
 Optimizations:
 - Only look at states in the rough open list
 - Check if a group of states can have successors in the current segment first
 - ?Only rank first 2 pieces first?
 
 */


