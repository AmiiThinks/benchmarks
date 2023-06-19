//
//  SimpleList.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 12/29/11.
//  Copyright (c) 2011 University of Denver. All rights reserved.
//

#ifndef Solve_Chinese_Checkers_SimpleList_h
#define Solve_Chinese_Checkers_SimpleList_h

#include <assert.h>

#define FREESIZE 5000+100

template <typename T>
class SimpleList {
	struct Node {
		Node() :next(0) {}
		Node( T val ) : value(val), next(0) { }
		void Init(T val) { value = val; next = 0; }
		T value;
		Node* next;
	};
	typedef Node* pNode;
	Node *freeList;
	Node freeData[FREESIZE];
	std::vector<Node *> q;

public:
	SimpleList(int theSize) {
		//freeData = new Node[1024*1024];
		freeList = &freeData[0];
		for (int x = 0; x < FREESIZE-1; x++)
			freeData[x].next = &freeData[x+1];
		q.reserve(theSize);
		q.resize(theSize);
	}
	~SimpleList() { /*delete [] freeData;*/ }
	int Insert(int index, T val)
	{
		if (q[index] && q[index]->value == val)
			return 0;
		if (freeList == 0)
		{
			assert(!"Free list too small");
			Node *tmp = new Node(val);
			tmp->next = q[index];
			q[index] = tmp;
		}
		else {
			Node *tmp = freeList;
			freeList = freeList->next;
			tmp->Init(val);
			tmp->next = q[index];
			q[index] = tmp;
		}
		return 1;
	}
	bool Empty(int index)
	{
		return q[index] == 0;
	}
	int MemorySize()
	{
		int cnt = 0;
		for (Node *t = freeList; t; t = t->next)
			cnt++;
		return cnt;
	}
	T Remove(int index)
	{
		assert(q[index]);
		T val = q[index]->value;
		Node *tmp = q[index];
		q[index] = q[index]->next;
		tmp->next = freeList;
		freeList = tmp;
		return val;
	}
	void Sort(int index)
	{
		if (!q[index]) return;
		q[index] = Sort(q[index]);
	}
private:
	Node *Sort(Node *n)
	{
		if (n == 0) return 0;
		if (n->next == 0) return n;
		pNode a, b;
		Split(n, a, b);
		a = Sort(a);
		b = Sort(b);
		return Merge(a, b);
	}
	void Split(pNode a, pNode &b, pNode &c)
	{
		b = 0;
		c = 0;
		while (a)
		{
			pNode t = a;
			a = a->next;
			t->next = b;
			b = t;
			if (!a) break;
			t = a;
			a = a->next;
			t->next = c;
			c = t;
		}
	}
	pNode Merge(pNode a, pNode b)
	{
		if (a == 0) return b;
		if (b == 0) return a;
		pNode result = 0;
		pNode tail;
		if (a->value < b->value)
		{
			result = a;
			a = a->next;
		}
		else {
			result = b;
			b = b->next;
		}
		tail = result;
		while (true)
		{
			if (a == 0)
			{
				tail->next = b;
				return result;
			}
			if (b == 0)
			{
				tail->next = a;
				return result;
			}
			if (a->value < b->value)
			{
				tail->next = a;
				a = a->next;
				tail = tail->next;
			}
			else {
				tail->next = b;
				b = b->next;
				tail = tail->next;
			}
		}
		assert(false);
		return 0;
	}
};

#endif
