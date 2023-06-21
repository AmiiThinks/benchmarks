#include <string>
#include <cassert>
#include <iostream>
#include <set>
#include <sstream>
#include "CCheckers.h"
#include "Timer.h"

//#undef CHECK_BOARDS 1
//#undef USE_V1_FUNCTIONS 1

using namespace std;

bool check_board(int* board, int toMove);
void TestRankUnrank();
void TestRankUnrank2();

// v1 rank / unrank functions
bool CCheckers__unrank_v1(int64_t theRank, CCState &s);
int64_t CCheckers__rank_v1(CCState &s);
int64_t CCheckers__getMaxRank_v1();

#ifdef USE_V1_FUNCTIONS
#define UNRANK CCheckers__unrank_v1
#define RANK CCheckers__rank_v1
#define GET_MAX_RANK CCheckers__getMaxRank_v1
#else
#define UNRANK cc.unrank
#define RANK cc.rank
#define GET_MAX_RANK cc.getMaxRank
#endif

void TestSymmetry();

int main (int argc, char * const argv[])
{
	//TestRankUnrank2();
	TestSymmetry();
	exit(0);
	
	
	CCheckers cc;
	CCState s;
	int ret = 0;
	
	set<string> boards;
	
	Timer t, overallT;
	overallT.StartTimer();
	t.StartTimer();
	
	int64_t maxRank = GET_MAX_RANK();
	int64_t fourPct = ((double)maxRank / 25.0);
	for (int64_t val = 0; val < maxRank; val++)
    {
		if (val && 0 == val%fourPct){
			t.EndTimer();
			cout << "Rank: " << val << " of " << maxRank
			<< " " << t.GetElapsedTime() << " seconds "
			<< (((double)fourPct) / t.GetElapsedTime()) << " states per second"
			<< endl;
			t.StartTimer();
		}
		
		// print out the ones with the error states
		if (!UNRANK(val, s))
		{
			// cout << "ERROR: unranking " << val << " failed" << endl;
			// s.PrintASCII();
			ret = 1;
		}
		else {
			if(RANK(s) != val){
				// cout << "ERROR: re-ranking " << val << " failed" << endl;
				// s.PrintASCII();
				ret = 1;
			}
#ifdef CHECK_BOARDS
			else {
				ostringstream oss;
				oss << s.toMove << " " << s.board;
				boards.insert(oss.str());
			}
#endif
		}
    }
	
#ifdef CHECK_BOARDS
	assert(boards.size() == maxRank);
#endif
	
	overallT.EndTimer();
	cout << "Overall Time : " << overallT.GetElapsedTime() << ", " << ((double)maxRank) / overallT.GetElapsedTime() << " states / second" << endl;
	return ret;
}

bool check_board(int* board, int toMove)
{
  int n1s = 0, n2s = 0;
  for(int i=0; i<NUM_SPOTS; ++i){
    switch(board[i]){
    case 2:
      n2s++;
      break;
    case 1:
      n1s++;
      break;
    case 0:
      break;
    default:
      assert("board has a value that is not 0, 1 or 2" == 0);
    }
  }
  return n1s == NUM_PIECES && n2s == NUM_PIECES && (toMove == 1 || toMove == 0);
}

int64_t CCheckers__getMaxRank_v1()
{
	int64_t total = 1;
	for (int x = 0; x < NUM_PLAYERS; x++)
	{
		for (int y = 0; y < NUM_PIECES; y++)
		{
			total *= NUM_SPOTS;
		}
	}
	return total*2; // who is first
}

int64_t CCheckers__rank_v1(CCState &s)
{
	int64_t theRank = s.toMove;
	for (int x = 0; x < NUM_PLAYERS; x++)
	{
		for (int y = 0; y < NUM_PIECES; y++)
		{
			theRank = theRank*NUM_SPOTS+s.pieces[x][y];
		}
	}
	return theRank;
}

bool CCheckers__unrank_v1(int64_t theRank, CCState &s)
{
	memset(s.board, 0, NUM_SPOTS*sizeof(int));	
	for (int x = NUM_PLAYERS-1; x >= 0; x--)
	{
		for (int y = NUM_PIECES-1; y >= 0; y--)
		{
			s.pieces[x][y] = theRank%NUM_SPOTS;
			if (s.board[s.pieces[x][y]] != 0)
				return false;
			s.board[s.pieces[x][y]] = x+1;
			theRank /= NUM_SPOTS;
		}
	}
	s.toMove = theRank;
	return true;
}

int64_t CurrentRank(CCState &s)
{
	int who = 0;
	int64_t r2 = 0;
	int last = NUM_SPOTS-1;
	for (int x = 0; x < NUM_PIECES; x++)
	{
		int64_t tmp = binomialSum(last, NUM_SPOTS-s.pieces[who][NUM_PIECES-1-x]-1, NUM_PIECES-1-x);
		//			printf("binomialSum(%d, %d, %d) = %llu\n", last, NUM_SPOTS-s.pieces[who][NUM_PIECES-1-x]-1, NUM_PIECES-1-x, tmp);
		r2 += tmp;
		last = NUM_SPOTS-s.pieces[who][NUM_PIECES-1-x]-1-1;
	}
	return r2;
}

int64_t SymmetricRank(CCheckers &cc, CCState &s)
{
	int64_t r2 = 0;
	int last = NUM_LEFT_SIDE-1;
	int who = 0;
	r2 = binomialSum(last, NUM_LEFT_SIDE-cc.getLeftID(s.pieces[who][NUM_PIECES-1])-1, NUM_PIECES-1-0);
	last = NUM_SPOTS-s.pieces[who][NUM_PIECES-1]-1-1;
	for (int x = 1; x < NUM_PIECES; x++)
	{
		int64_t tmp = binomialSum(last, NUM_SPOTS-s.pieces[who][NUM_PIECES-1-x]-1, NUM_PIECES-1-x);
		//			printf("binomialSum(%d, %d, %d) = %llu\n", last, NUM_SPOTS-s.pieces[who][NUM_PIECES-1-x]-1, NUM_PIECES-1-x, tmp);
		r2 += tmp;
		last = NUM_SPOTS-s.pieces[who][NUM_PIECES-1-x]-1-1;
	}
	return r2;
}

void TestSymmetry()
{
	CCheckers cc;
	CCState s;
	//const int64_t maxBucketSize = 16000000000; // 16 billion entries = 2 billion bytes (2GB file size)
	//const int64_t maxBucketSize = 64000000000; // 64 billion entries = 8 billion bytes (8GB RAM)
	//const int64_t maxBucketSize = 256000000000; // 256 billion entries = 32 billion bytes (32GB RAM) [25%]
	const int64_t maxBucketSize = 541165879296ull;
	int64_t maxRank = cc.getMaxSinglePlayerRank();
	std::vector<int> bucket;
	std::vector<int64_t> bucketSize;
	int currBucket = 0;
	bucketSize.push_back(0);
	//std::vector<bool> ranks(maxRank);
	int64_t used = 0;
	for (int64_t x = 0; x < maxRank; x++)
	{
		bool flag = false;
		int64_t r1, r2;
		cc.unrankPlayer(x, s, 0);
		if (!cc.symmetricStart(s))
		{
			//used++;
			flag = true;
		}
		
		cc.rankPlayer(s, 0, r1, r2);
		int64_t maxVal = cc.getMaxSinglePlayerRank2(r1);
		int64_t maxVal2 = cc.getMaxSinglePlayerRank2();
		x+=maxVal-1;
		if (!flag)
		{
			//s.Print();
			printf("%llu : %llu of %llu; %llu of %llu [*]\n", x, r1, maxVal2, r2, maxVal);
			bucket.push_back(-1);
		}
		else {
			if (bucketSize[currBucket] + maxVal > maxBucketSize)
			{
				currBucket++;
				bucketSize.push_back(0);
			}
			bucketSize[currBucket] += maxVal;
			bucket.push_back(currBucket);
			printf("%llu : %llu of %llu; %llu of %llu [Bucket %d]\n", x, r1, maxVal2, r2, maxVal, currBucket);
			used += maxVal;
			
		}
	}
	printf("%llu of %llu entries needed; %ld buckets\n", used, maxRank, bucketSize.size());
	for (unsigned int x = 0; x < bucketSize.size(); x++)
	{
		printf("Bucket %d size %llu [%1.2f GB]\n", x, bucketSize[x], (float)bucketSize[x]/1024.0/1024.0/1024.0/8.0);
	}
	printf("Done! no errors\n");
}

void TestRankUnrank()
{
	CCheckers cc;
	CCState s;
	
	int64_t maxSinglePlayerRank;

	maxSinglePlayerRank = cc.getMaxSinglePlayerRank();

	int64_t r;
	for (int64_t x = 0; x < maxSinglePlayerRank; x++)
	{
		cc.unrankPlayer(x, s, 0);
		r = cc.rankPlayer(s, 0);
		if (r != x)
		{
			s.Print();
			printf("Error; %llu ranked into above state, but unranked into %llu\n", x, r);
			exit(0);
		}
	}
}

void TestRankUnrank2()
{
	CCheckers cc;
	CCState s, t;
	
	int64_t maxRank;
	
	maxRank = cc.getMaxRank();
	
	int64_t r;
	for (int64_t x = 0; x < maxRank; x++)
	{
		cc.unrank(x, s);
		r = cc.rankPlayer(s, 0);
		cc.unrankPlayer(r, t, 0);
		for (int y = 0; y < NUM_PIECES; y++)
		{
			assert(s.pieces[0][y] == t.pieces[0][y]);
		}
//		if (r != x)
//		{
//			s.Print();
//			printf("Error; %llu ranked into above state, but unranked into %llu\n", x, r);
//			exit(0);
//		}
	}
}

void TestRankUnrank3()
{
	CCheckers cc;
	CCState s;
	cc.Reset(s);
	s.Print();
	int64_t startRank = cc.rankPlayer(s, 0);
	printf("(%llu)-------\n", startRank);
//	//	cc.unrankPlayer(startRank, s, 0);
//	//	s.Print();
//	//	printf("-------\n");
//	s.Reverse();
//	s.Print();
//	printf("-------\n");
//	int64_t goal = cc.rankPlayer(s, 0);
//	printf("Starting at goal %llu dist %d\n", goal, startDist[goal]);
}