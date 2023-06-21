//
//  LockFreeQueue.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 10/26/11.
//  Copyright (c) 2011 University of Denver. All rights reserved.
//

#ifndef Solve_Chinese_Checkers_LockFreeQueue_h
#define Solve_Chinese_Checkers_LockFreeQueue_h

// Implementation from http://drdobbs.com/high-performance-computing/210604448
// Note that the code in that article (10/26/11) is broken.
// The attempted fixed version is below.
#include <pthread.h>

template <typename T>
class LockFreeQueue {
private:
	struct Node {
		Node( T val ) : value(val), next(0) { }
		T value;
		Node* next;
	};
	int entries;
	Node *first,      // for producer only
	*divider, *last;    // shared
	//pthread_mutex_t lock;
public:
	LockFreeQueue()
	{
		first = divider = last = new Node(T()); // add dummy separator
		//printf("alloc1 %p\n", first);
		//pthread_mutex_init(&lock, NULL);
	}
	~LockFreeQueue()
	{
		//pthread_mutex_lock (&lock);
		while( first != 0 )    // release the list
		{
			Node* tmp = first;
			first = tmp->next;
			delete tmp;
		}
		////pthread_mutex_unlock (&lock);
	}
	bool Empty() { return divider==last; }
	void Produce( const T& t )
	{
		//pthread_mutex_lock (&lock);
		last->next = new Node(t);    // add the new item
		entries++;
		//printf("alloc2 %p\n", last->next);
		last = last->next;      // publish it

		while (first != divider) // trim unused nodes
		{
			Node* tmp = first;
			first = first->next;
			//printf("free %p\n", tmp);
			delete tmp;
		}
		//pthread_mutex_unlock (&lock);
	}
	bool Consume( T& result )
	{
		//pthread_mutex_lock (&lock);
		if (divider != last)         // if queue is nonempty
		{
			result = divider->next->value; // C: copy it back
			divider = divider->next;      // D: publish that we took it
			entries--;
			//pthread_mutex_unlock (&lock);
			return true;                  // and report success
		}
		//pthread_mutex_unlock (&lock);
		return false;                   // else report empty
	}
	int GetMemory()
	{ return 0; }
	int NumEntries() { return entries; }
};

#endif
