//
//  LockQueue.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 10/26/11.
//  Copyright (c) 2011 University of Denver. All rights reserved.
//

#ifndef Solve_Chinese_Checkers_LockQueue_h
#define Solve_Chinese_Checkers_LockQueue_h

#include <pthread.h>
#include <assert.h>

const int lockQueueSize = 256;

template <typename T>
class LockQueue {
private:
	struct Node {
		Node():next(0) {}
		Node( T val ) : value(val), next(0) { }
		void Init(T val) { value = val; next = 0; }
		T value;
		Node* next;
	};
	Node *first,      // for producer only
	*last;    // shared
	Node *freeList;
	Node freeData[lockQueueSize];
	pthread_mutex_t lock;
    pthread_cond_t waitConsume;
    pthread_cond_t waitProduce;
	int entries;
public:
	LockQueue()
	{
		freeList = &freeData[1];
		for (int x = 1; x < lockQueueSize-1; x++)
			freeData[x].next = &freeData[x+1];
		first = last = &freeData[0]; // add dummy separator
		//printf("alloc1 %p\n", first);
		pthread_mutex_init(&lock, NULL);
		pthread_cond_init(&waitConsume, NULL);
		pthread_cond_init(&waitProduce, NULL);
		entries = 0;
	}
	~LockQueue()
	{
		pthread_mutex_lock (&lock);
//		while( first != 0 )    // release the list
//		{
//			Node* tmp = first;
//			first = tmp->next;
//			//printf("free1 %p\n", tmp);
//			delete tmp;
//		}
		pthread_mutex_unlock (&lock);
	}
	int GetMemory()
	{
		int cnt = 0;
		for (Node *tmp = freeList; tmp; tmp = tmp->next)
			cnt++;
		return cnt;
	}
	bool Empty() { return (entries==0); }
	int NumEntries() { return entries; }
	void Produce( const T& t )
	{
		pthread_mutex_lock (&lock);
		while (freeList == 0)// || entries > lockQueueSize)
		{
			pthread_cond_wait(&waitProduce, &lock);
		}
		
		if (freeList == 0)
		{
			assert(!"LockQueue pool too small");
			last->next = new Node(t);    // add the new item
		}
		else {
			last->next = freeList;
			freeList = freeList->next;
			last->next->Init(t);
		}
		//printf("alloc2 %p\n", last->next);
		last = last->next;      // publish it
		entries++;
		pthread_mutex_unlock (&lock);
		pthread_cond_signal(&waitConsume);
	}
	bool Consume( T& result )
	{
		pthread_mutex_lock (&lock);
		while (first == last)
		{
			pthread_cond_wait(&waitConsume, &lock);
		}
		if (first != last)
			// if queue is nonempty
		{
			result = first->next->value; // C: copy it back
			Node *tmp = first;
			first = first->next;
			//printf("free2 %p\n", tmp);
			tmp->next = freeList;
			freeList = tmp;
			//delete tmp;
			entries--;
			pthread_mutex_unlock (&lock);
			pthread_cond_signal(&waitProduce);
			return true;                  // and report success
		}
		pthread_mutex_unlock (&lock);
		return false;                   // else report empty
	}
};

#endif
