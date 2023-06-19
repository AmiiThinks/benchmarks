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
int THREADS;
void *Worker(void *division);
pthread_mutex_t lock;


int main(int argc, char * const argv[])
{
	Timer t;
	t.StartTimer();
	FindMaxMinSuccessors(atoi(argv[1]));
	t.EndTimer();
	printf("%1.3f seconds elapsed\n", t.GetElapsedTime());
	return 0;
	//TestSuccessors();
	//return 0;
	
	//SimpleSolveGame();
	std::vector<bool> proof;
	PrioritySolver(proof, "4-piece-proof");
	//PrioritySolver();
//	if (argc > 1)
//		SolveWithThreads(atoi(argv[1]));
//	else
//		SolveWithThreads(1);
	
	t.EndTimer();
	printf("%1.3f seconds elapsed\n", t.GetElapsedTime());
	
	return 0;
	
	
//	if (argc > 1)
//		SolveWithThreads(atoi(argv[1]));
//	else
//		SolveWithThreads(1);

    return 0;
}

void TestSuccessors()
{
	CCheckers cc;
	CCState s;
	cc.Reset(s);
	s.Print();
	
	while (!cc.Done(s))
	{
		CCMove *m, *n;
		n = m = cc.getMoves(s);
		assert(m != 0);
		
		int cnt = 1; // choose a random move
		while (m->next != 0)
		{
			if (0 == random()%cnt)
			{
				cnt++;
				m = m->next;
			}
			else {
				break;
			}
		}
		cc.ApplyMove(s, m);
		s.Print();
		cc.freeMove(n);
	}
	getchar();
}


void FindMaxMinSuccessors(int numThreads)
{
	CCheckers cc;
	CCState s;
	cc.getMaxSinglePlayerRank();
	cc.unrankPlayer(0, s, 0);
	THREADS = numThreads;
	pthread_mutex_init(&lock, NULL);
	printf("Creating %d threads\n", THREADS);
	sleep(1);
	std::vector<pthread_t> threads;
	threads.resize(numThreads);
	for (int x = 0; x < numThreads; x++)
	{
		pthread_create(&threads[x], NULL, Worker, (void *)x);
	}
	for (int x = 0; x < numThreads; x++)
		pthread_join(threads[x], 0);
}

void *Worker(void *division)
{
	int t = THREADS;
	int min = 1000, max = 0;
	uint64_t minRank, maxRank;
	uint64_t tot = 0;
	CCheckers cc;
	CCState s;
	long cnt = (long)division;
	uint64_t numStates = cc.getMaxSinglePlayerRank();
	//printf("Thread %d of %d started. %llu total states to unrank\n", cnt, t, numStates);
	for (uint64_t x = cnt; x < numStates; x+=THREADS)
	  //for (uint64_t x = numStates-1; x > 0; x--)
	{
	  //if ((x%t) == cnt)
		{
			tot++;
			continue;
			cc.unrankPlayer(x, s, 0);
			//if (cc.symmetricStart(s))
			//	continue;
			CCMove *m = cc.getMoves(s);
			int len = m->length();
			cc.freeMove(m);
			if (len < min)
			{
				min = len;
				minRank = x;
				/*
				pthread_mutex_lock (&lock);
				printf("[%ld] Min rank %llu has %d moves\n", cnt, minRank, min);
				s.PrintASCII();
				fflush(stdout);
				pthread_mutex_unlock (&lock);
				*/
			}
			if (len > max)
			{
				max = len;
				maxRank = x;
				/*
				pthread_mutex_lock (&lock);
				printf("[%ld] Max rank %llu has %d moves\n", cnt, maxRank, max);
				s.PrintASCII();
				fflush(stdout);
				pthread_mutex_unlock (&lock);
				*/
			}
		}
	}
	printf("Thread %d finished. %llu unrankings\n", cnt, tot);
	/*	
		pthread_mutex_lock (&lock);
	
	printf("[%ld] Min rank %llu has %d moves\n", cnt, minRank, min);
	printf("[%ld] Max rank %llu has %d moves\n", cnt, maxRank, max);
	//cc.unrankPlayer(minRank, s, 0);
	//s.Print();
	//s.PrintASCII();
	//cc.unrankPlayer(maxRank, s, 0);
	//s.Print();
	//s.PrintASCII();
	pthread_mutex_unlock (&lock);
	*/
	pthread_exit(0);
}
