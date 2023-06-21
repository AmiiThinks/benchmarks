#include <cassert>
#include <deque>
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <sstream>
#include <stdio.h>
#include "CCheckers.h"
#include "test_utils.h"

int64_t binomial(unsigned int n, unsigned int k);
void test1();
void test2();

using namespace std;

int main(int argc, char** argv)
{
	test1();
	// test2();
}

void test1()
{
	CCheckers cc;
	CCState s;
	uint64_t val2;
	set<string> boards;

	uint64_t maxRank = cc.getMaxSinglePlayerRank();
	cc.Reset(s);
	// player 1
	for (uint64_t val = 0; val < maxRank; val++)
	{
		assert(cc.unrankPlayer(val, s, 0));
		val2 = cc.rankPlayer(s, 0);
		assert(val == val2);

		ostringstream oss;
		oss << s.board;
		boards.insert(oss.str());

		cout << val << " unranks to " << oss.str() << endl;
	}
	assert(boards.size() == maxRank);

	boards.clear();

	// player 2
	for (uint64_t val = 0; val < maxRank; val++)
	{
		assert(cc.unrankPlayer(val, s, 1));
		val2 = cc.rankPlayer(s, 1);
		assert(val == val2);

		ostringstream oss;
		oss << s.board;
		boards.insert(oss.str());
		cout << val << " unranks to " << oss.str() << endl;
	}
	assert(boards.size() == maxRank);
}

void test2()
{
	CCheckers cc;
	CCState s;

	std::deque<uint64_t> q;
	std::vector<uint8_t> d;
	uint64_t maxRank = cc.getMaxSinglePlayerRank();
	d.resize(maxRank);
	cc.Reset(s);
	for (uint64_t val = 0; val < maxRank; val++)
	{
		d[val] = 255;
	}
	q.push_back(cc.rankPlayer(s, 0));
	d[cc.rankPlayer(s, 0)] = 0;
	int tmp = s.pieces[0][0];
	s.pieces[0][0] = s.pieces[0][1];
	s.pieces[0][1] = tmp;
	q.push_back(cc.rankPlayer(s, 0));
	d[cc.rankPlayer(s, 0)] = 0;
//		if (cc.unrankPlayer(val, s, 0))
//		{
//			if (cc.Done(s))
//			{
//				q.push_back(val);
//				d[val] = 0;
//			}
//		}
//	}
	printf("%d goal states", (int)q.size());
	
	uint64_t next;
	while (q.size() > 0)
	{
		next = q.front();
		q.pop_front();
		
		cc.unrankPlayer(next, s, 0);
		CCMove *m = cc.getMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyMove(s, t);
			uint64_t child = cc.rankPlayer(s, 0);
			if (d[child] == 255)
			{
				d[child] = d[next]+1;
				q.push_back(child);
			}
			cc.UndoMove(s, t);
		}
	}
	cc.unrankPlayer(next, s, 0);
	s.Print();
	printf("Farthest state\n");
	
	std::vector<int> counts;
	counts.resize(255);
	for (unsigned int x = 0; x < d.size(); x++)
		counts[d[x]]++;
	for (unsigned int x = 0; x < counts.size(); x++)
		printf("%d:: %d\n", x, counts[x]);
}
