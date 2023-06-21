//
//  Memory.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 12/16/14.
//
//

#include "Memory.h"
#include <assert.h>
#include <algorithm>

static bool debug = false;

Memory::Memory()
{
	head.entries = 1ull<<40ull; //1024 TB?
	head.next = 0;
	head.prev = 0;
	head.free = true;
	maxUsage = 0;
	currUsage = 0;
}

size_t Memory::Alloc(size_t bytes)
{
	currUsage += bytes;

	size_t offset = 0;
	for (freeList *t = &head; t; t = t->next)
	{
		if (t->free && t->entries > bytes)
		{
			freeList *tmp = new freeList;
			tmp->next = t->next;
			tmp->prev = t;
			if (t->next)
			{
				t->next->prev = tmp;
			}
			t->next = tmp;
			tmp->entries = t->entries-bytes;
			t->entries = bytes;

			t->free = false;
			tmp->free = true;
			maxUsage = std::max(maxUsage, t->entries+offset);
			
			if (debug) { Print(); }
			return offset;
		}
		if (t->free && t->entries == bytes)
		{
			t->free = false;
			if (debug) { Print(); }
			return offset;
		}
		offset += t->entries;
	}
	if (debug) { Print(); }
	assert(!"No memory available");
	return -1;
}

void Memory::Free(size_t location)
{
	size_t offset = 0;
	for (freeList *t = &head; t; t = t->next)
	{
		if (offset == location)
		{
			currUsage -= t->entries;
			
			assert(t->free == false);
			// can merge with previous
			if (t->prev && t->prev->free)
			{
				freeList *tmp = t;
				t = t->prev;
				t->entries += tmp->entries;
				t->next = tmp->next;
				if (tmp->next)
				{
					tmp->next->prev = t;
				}
				delete tmp;
			}
			// can merge with next
			if (t->next && t->next->free)
			{
				freeList *tmp = t->next;
				t->entries += tmp->entries;
				t->next = tmp->next;
				if (tmp->next)
				{
					tmp->next->prev = t;
				}
				delete tmp;
			}
			t->free = true;
			//if (debug) { Print(); }
			return;
		}
		offset += t->entries;
	}
	if (debug) { Print(); }
	assert(!"Memory not found to free!");
}

void Memory::Print()
{
	printf("-->\n");
	size_t offset = 0;
	for (freeList *t = &head; t; t = t->next)
	{
		printf("At offset %lu, memory size %lu is %s\n", offset, t->entries, t->free?"free":"in use");
		offset += t->entries;
	}
	printf("<-- (Curr: %lu; High Water: %lu)\n", currUsage, maxUsage);
}

//private:
//struct freeList {
//	size_t entries;
//	bool free;
//	freeList *prev;
//	freeList *next;
//};
//freeList head;
