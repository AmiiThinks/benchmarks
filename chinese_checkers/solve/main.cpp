//
//  main.cpp
//  CC Solver
//
//  Created by Nathan Sturtevant on 4/22/16.
//  Copyright © 2016 NS Software. All rights reserved.
//

#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include "CCUtils.h"
#include "ForwardSolver.h"
#include "BackwardSolver.h"
#include "CCheckers.h"
#include "BackwardBitSolver.h"
#include "HybridBitSolver.h"
#include "BaselineSolver.h"
#include "OptimizedSolver.h"
#include "CCRankings.h"
#include "SymmetrySolver.h"
#include "LegacyTests.h"
#include "RankingTests.h"
#include "CacheFriendlySolver.h"
#include "FullSymmetrySolver.h"
#include "InitialParallelSolver.h"
#include "FasterSymmetrySolver.h"
#include "EMSolver.h"
#include "ParallelEMSolver.h"
#include "SimpleP1P2Solver.h"

std::string gFilePath;
//char gFilePath[2048];// = "/Users/nathanst/Desktop/CC/";


void CompareBitToBaseline();
void CompareOptimizedToBaseline();
void CompareRankingResults();
void VerifySymmetryCorrectness();
void TestSymmetrySolver();
void BigSolveForLaptop();
void LocalityTest();
void TestCacheSolver();
void TestFullSymmetrySolver();
void ParallelTest();
void ParallelEMTest(bool resume);
void EMTest();
void ExploreProof();
void PrintStats();

void SaveAsSVG(const CCState &s);
void MakeLocality();
void P1P2Test();

int main(int argc, const char * argv[]) {
	setvbuf(stdout, NULL, _IONBF, 0);

	// Show the initial board
//	{
//		CCState s;
//		CCheckers cc;
//		cc.Reset(s);
//		SaveAsSVG(s);
//		exit(0);
//	}
	
	// show the order of the ranking
	if (0)
	{
		{
			CCLocationColorRank r;
			//		CCPSRank12 r;
			CCState s;
			for (int x = 0; x < 10; x++)
			{
				r.unrank(x, s);
				SaveAsSVG(s);
			}
			uint64_t rank = ConstChoose(NUM_PIECES*2, NUM_PIECES);
			for (int x = 0; x < 10; x++)
			{
				r.unrank(rank, s);
				SaveAsSVG(s);
				rank += ConstChoose(NUM_PIECES*2, NUM_PIECES);
			}
		}

		{
			CCColorLocationRank r;
			//		CCPSRank12 r;
			CCState s;
			for (int x = 0; x < 10; x++)
			{
				r.unrank(x, s);
				SaveAsSVG(s);
			}
			uint64_t rank = ConstChoose(NUM_SPOTS, 2*NUM_PIECES);
			for (int x = 0; x < 10; x++)
			{
				r.unrank(rank, s);
				SaveAsSVG(s);
				rank += ConstChoose(NUM_SPOTS, 2*NUM_PIECES);
			}
		}

		{
			CCPSRank12 r;
			CCState s;
			for (int x = 0; x < 10; x++)
			{
				r.unrank(x, s);
				SaveAsSVG(s);
			}
			uint64_t rank = ConstChoose(NUM_SPOTS-NUM_PIECES, NUM_PIECES);
			for (int x = 0; x < 10; x++)
			{
				r.unrank(rank, s);
				SaveAsSVG(s);
				rank += ConstChoose(NUM_SPOTS-NUM_PIECES, NUM_PIECES);
			}
			
		}

		{
			CCDefaultRank r;
			CCState s;
			for (int x = 0; x < 20; x++)
			{
				r.unrank(x, s);
				SaveAsSVG(s);
			}
			for (int x = 20; x < 2000; x+=20)
			{
				r.unrank(x, s);
				SaveAsSVG(s);
			}
//			uint64_t rank = ConstChoose(NUM_SPOTS-NUM_PIECES, NUM_PIECES);
//			for (int x = 0; x < 10; x++)
//			{
//				r.unrank(rank, s);
//				SaveAsSVG(s);
//				rank += ConstChoose(NUM_SPOTS-NUM_PIECES, NUM_PIECES);
//			}
		}
		
		exit(0);
	}

	// show symmetry
//	{
//		CCheckers cc;
//		CCPSRank12 r;
//		CCState s;
//		r.unrank(random()%r.getMaxRank(), s);
//		SaveAsSVG(s);
//		cc.SymmetryFlipHoriz(s);
//		SaveAsSVG(s);
//		cc.SymmetryFlipVert(s);
//		SaveAsSVG(s);
//		exit(0);
//	}

//	{
//		srandom(66);
//		CCheckers cc;
//		CCPSRank12 r;
//		CCState s;
//		r.unrank(random()%r.getMaxRank(), s);
//		SaveAsSVG(s);
//		CCMove *m = cc.getReverseMoves(s);
//		for (CCMove *n = m; n; n = n->next)
//		{
//			cc.ApplyReverseMove(s, n);
//			SaveAsSVG(s);
//			cc.SymmetryFlipVert(s);
//			SaveAsSVG(s);
//			cc.SymmetryFlipVert(s);
//			cc.UndoReverseMove(s, n);
//		}
//		exit(0);
//	}
//
	
//	// Initial states
//	{
//		CCState s;
//		CCPSRank12 r;
//
//		for (int64_t x = 0; x < 10; x++)
//		{
//			r.unrankP1(x, s);
//			SaveAsSVG(s);
//		}
//		for (int64_t x = 10; x < r.getMaxP1Rank(); x+=1000)
//		{
//			r.unrankP1(x, s);
//			SaveAsSVG(s);
//		}
//		exit(0);
//	}
	
	if (argc < 3)
	{
		printf("Usage: %s -path <path_to_data>\n", argv[0]);
		exit(0);
	}
	gFilePath = argv[2];

	for (int x = 0; x < argc; x++)
	{
		if (strcmp("-stats", argv[x]) == 0)
		{
			PrintStats();
			exit(0);
		}
	}

	for (int x = 0; x < argc; x++)
	{
		if (strcmp("-play", argv[x]) == 0)
		{
			ExploreProof();
			exit(0);
		}
	}
	
//	for (int x = 0; x < 100; x++)
//	{
//		CCheckers cc;
//		CCState s;
//		cc.unrankPlayer(x, s, 0);
//		s.Print();
//		s.PrintASCII();
//		getchar();
//		cc.SymmetryFlipVertP1(s);
//		s.Print();
//		s.PrintASCII();
//		printf("---\n\n");
//		getchar();
//	}
//	exit(0);
	
//	for (int p = 1; p <= 6; p++)
//		printf("%d locs %d pieces, %llu entries per group, %llu groups\n",
//			   NUM_SPOTS, p,
//			   ConstChoose(NUM_SPOTS-p, p),
//			   ConstGetSymmetricP1Ranks(DIAGONAL_LEN, p));
	
	
	// TODO: here
	//P1P2Test();
	
	
	ParallelEMTest(false);
	//EMTest();
	
	//TestFullSymmetrySolver();
	//ParallelTest();
	//TestCacheSolver();
	
//	MakeLocality();
//	LocalityTest();
	//TestRankingCorrectness<CCPSRank12a>();
	//TestMapping<CCPSRank12a, CCPSRank12>();
	//TestRanking();
	//TestRankingSpeed();
//	TestSymmetrySolver();
//	BigSolveForLaptop();
//	TestChildLocality<CCPSRank12>();
//	TestChildLocality<CCPSRank21>();
//	TestChildLocality<CCPSRank21a>();
//	TestChildLocality<CCPSRank12a>();
	return 0;
}

void P1P2Test()
{
	CCState s;
	CCheckers cc;
	CCLocalRank12 l;
	
	if (0)
	{
		uint64_t which = 10;
		CCPSRank12 r;
		uint64_t r2 = r.getMaxP2Rank();
		for (uint64_t x = 0; x < r2; x++)
		{
			r.unrank(which, x, s);
			s.toMove = 0;
			printf("p1p2: %6llu %6llu", r.rankP1(s), r.rankP2(s));
			s.toMove = 1;
			printf(" | p2p1: %6llu %6llu ", r.rankP1(s), r.rankP2(s));
			s.PrintASCII();
			
			CCMove *m = cc.getMoves(s);
			for (CCMove *tmp = m; tmp; tmp = tmp->next)
			{
				cc.ApplyMove(s, tmp);

				printf(">p2p1: %6llu %6llu  ", r.rankP1(s), r.rankP2(s));
				s.toMove = 1;
				printf(" | p1p2: %6llu %6llu", r.rankP1(s), r.rankP2(s));
				s.toMove = 0;
				s.PrintASCII();

				cc.UndoMove(s, tmp);
			}
			cc.freeMove(m);
		}
		exit(0);
	}
	
	// NOTE: Number of threads set in header file. Code can be more efficient this way.
	SimpleP1P2Solver::Solver *p1p2 = new SimpleP1P2Solver::Solver(gFilePath.c_str(), gFilePath.c_str()); // true - to clear data for build
	SimpleP1P2Solver::Solver &p1p2s = *p1p2;
	if (p1p2s.NeedsBuild())
		p1p2s.BuildData();
	
	FasterSymmetry::Solver base(gFilePath.c_str(), true, false);// optimize order=true; rebuild=false by default
	//CCBaselineSolver::BaselineSolver base(gFilePath.c_str(), true);
	
	for (int64_t x = 0; x < l.getMaxRank(); x++)
	{
		l.unrank(x, s);
		if (base.Lookup(s) != p1p2s.Lookup(s))
		{
			printf("Current state\n");
			s.PrintASCII();
			{
				CCMove *c = cc.getMoves(s);
				for (CCMove *n = c; n; n = n->next)
				{
					cc.ApplyMove(s, n);
					printf("{base:%s} ", FasterSymmetry::resultText[base.Lookup(s)] );
					s.PrintASCII();
					printf("{p1p2:%s} ", FasterSymmetry::resultText[p1p2s.Lookup(s)] );
					s.PrintASCII();
					cc.UndoMove(s, n);
				}
				
			}
			printf("State %llu has %s (old) %s (new code)\n", x, FasterSymmetry::resultText[base.Lookup(s)], FasterSymmetry::resultText[p1p2s.Lookup(s)]);
			//printf("State %llu has %d (old) %d (new code)\n", x, sym2.Lookup(s), ems.Lookup(s));
			printf("FAILURE\n");
			exit(0);
		}
	}
	printf("SUCCESS\n");
	exit(0);
}

void ParallelEMTest(bool resume)
{
	CCState s;
	CCheckers cc;
	CCLocalRank12 l;
	
	// NOTE: Number of threads set in header file. Code can be more efficient this way.
	ParallelEM::Solver *emsp = new ParallelEM::Solver(gFilePath.c_str(), gFilePath.c_str(), resume==false); // true - to clear data for build
	ParallelEM::Solver &ems = *emsp;
	if (ems.NeedsBuild())
		ems.BuildData();
	
	FasterSymmetry::Solver base(gFilePath.c_str(), true, false);// optimize order=true; rebuild=false by default
	base.PrintStats();

//	CCBaselineSolver::BaselineSolver base(gFilePath.c_str(), true);

	for (int64_t x = 0; x < l.getMaxRank(); x++)
	{
		l.unrank(x, s);
		if (base.Lookup(s) != ems.Lookup(s))
		{
			printf("Current state\n");
			s.PrintASCII();
			{
				CCMove *c = cc.getMoves(s);
				for (CCMove *n = c; n; n = n->next)
				{
					cc.ApplyMove(s, n);
					printf("{base:%s} ", FasterSymmetry::resultText[base.Lookup(s)] );
					s.PrintASCII();
					printf("{ems:%s} ", FasterSymmetry::resultText[ems.Lookup(s)] );
					s.PrintASCII();
					cc.UndoMove(s, n);
				}

			}
			printf("State %llu has %s (old) %s (new code)\n", x, FasterSymmetry::resultText[base.Lookup(s)], FasterSymmetry::resultText[ems.Lookup(s)]);
			//printf("State %llu has %d (old) %d (new code)\n", x, sym2.Lookup(s), ems.Lookup(s));
			printf("FAILURE\n");
			exit(0);
		}
	}
	printf("SUCCESS\n");
	exit(0);
}

void EMTest()
{
	CCState s;
	CCheckers cc;
	CCLocalRank12 l;

	EM::Solver *emsp = new EM::Solver(gFilePath.c_str(), gFilePath.c_str(), true); // true - to clear data for build
	EM::Solver &ems = *emsp;
	if (ems.NeedsBuild())
		ems.BuildData();

	FasterSymmetry::Solver sym(gFilePath.c_str(), true);// optimize order=true; rebuild=false by default
	sym.PrintStats();
	CCBaselineSolver::BaselineSolver base(gFilePath.c_str(), false);

	for (int64_t x = 0; x < l.getMaxRank(); x++)
	{
		l.unrank(x, s);
		if (base.Lookup(s) == CCBaselineSolver::kIllegal && ems.Lookup(s) != EM::kIllegal)
		{
			s.PrintASCII();
			printf("State %llu has %s (ems) %s (base)\n", x, FasterSymmetry::resultText[ems.Lookup(s)], FasterSymmetry::resultText[base.Lookup(s)]);
			printf("FAILURE: Children:\n");
			{
				CCMove *c = cc.getMoves(s);
				for (CCMove *n = c; n; n = n->next)
				{
					cc.ApplyMove(s, n);
					printf(" {sym:%s} ", FasterSymmetry::resultText[ems.Lookup(s)] );
					s.PrintASCII();
					printf(" {bas:%s} ", FasterSymmetry::resultText[base.Lookup(s)] );
					s.PrintASCII();
					cc.UndoMove(s, n);
				}
				cc.freeMove(c);
			}
			exit(0);
		}
	}
	printf("PASS: All illegals match\n");
	for (int64_t x = 0; x < l.getMaxRank(); x++)
	{
		l.unrank(x, s);
		if (base.Lookup(s) != ems.Lookup(s))
		{
			printf("Current state [%d]:", cc.Winner(s));
			s.PrintASCII();
			{
				CCMove *c = cc.getMoves(s);
				for (CCMove *n = c; n; n = n->next)
				{
					cc.ApplyMove(s, n);
					printf("{sym:%s} ", FasterSymmetry::resultText[base.Lookup(s)] );
					s.PrintASCII();
					printf("{ems:%s} ", ems.LookupText(s) );
					s.PrintASCII();
					cc.UndoMove(s, n);
				}
				cc.freeMove(c);
			}
			printf("State %llu has %s (old) %s (new code)\n", x, FasterSymmetry::resultText[base.Lookup(s)], ems.LookupText(s));
			printf("FAILURE\n");
			exit(0);
		}
	}
	printf("SUCCESS\n");

}

void ParallelTest()
{
	CCLocalRank12 l;
	CCheckers cc;
	CCState s;
	
	ParallelOne::Solver p(gFilePath.c_str(), true);
	CCSymmetrySolver::SymmetrySolver<CCPSRank21a, false> sym21a(gFilePath.c_str());
	
	for (int64_t x = 0; x < l.getMaxRank(); x++)
	{
		l.unrank(x, s);
		if (sym21a.Lookup(s) != p.Lookup(s))
		{
			s.PrintASCII();
			printf("State %llu has %d (old) %d (new code)\n", x, sym21a.Lookup(s), p.Lookup(s));
			printf("FAILURE\n");
			exit(0);
		}
	}
	printf("SUCCESS\n");
	
	
	exit(0);
}

void TestFullSymmetrySolver()
{
	CCLocalRank12 l;
	CCheckers cc;
	CCState s;
	
	CCBaselineSolver::BaselineSolver base(gFilePath.c_str(), false);
//	FasterSymmetry::Solver sym(gFilePath.c_str(), true, true);
	FullSymmetry::Solver sym2(gFilePath.c_str(), true, true);
//	CCSymmetrySolver::SymmetrySolver<CCPSRank21a, false> sym21a(gFilePath.c_str());

	// First look for illegal mismatches
	for (int64_t x = 0; x < l.getMaxRank(); x++)
	{
		l.unrank(x, s);
		if (base.Lookup(s) == CCBaselineSolver::kIllegal && sym2.Lookup(s) != FasterSymmetry::kIllegal)
		{
			s.PrintASCII();
			printf("State %llu has %s (sym2) %s (base)\n", x, FasterSymmetry::resultText[sym2.Lookup(s)], FasterSymmetry::resultText[base.Lookup(s)]);
			printf("FAILURE: Children:\n");
			{
				CCMove *c = cc.getMoves(s);
				for (CCMove *n = c; n; n = n->next)
				{
					cc.ApplyMove(s, n);
					printf(" {sym:%s} ", FasterSymmetry::resultText[sym2.Lookup(s)] );
					s.PrintASCII();
					printf(" {bas:%s} ", FasterSymmetry::resultText[base.Lookup(s)] );
					s.PrintASCII();
					cc.UndoMove(s, n);
				}
				cc.freeMove(c);
			}
			exit(0);
		}
	}
	printf("PASS: All illegals match\n");
	// Then look for other mismatches
	for (int64_t x = 0; x < l.getMaxRank(); x++)
	{
		l.unrank(x, s);
		if (sym2.Lookup(s) != base.Lookup(s))
		{
			s.PrintASCII();
			printf("State %llu has %s (sym2) %s (base)\n", x, FasterSymmetry::resultText[sym2.Lookup(s)], FasterSymmetry::resultText[base.Lookup(s)]);
			printf("FAILURE: Children:\n");
			{
				CCMove *c = cc.getMoves(s);
				for (CCMove *n = c; n; n = n->next)
				{
					cc.ApplyMove(s, n);
					printf(" {sym:%s}[%d] ", FasterSymmetry::resultText[sym2.Lookup(s)], cc.Winner(s) );
					s.PrintASCII();
					printf(" [%llu]{bas:%s}[%d] ", cc.rank(s), FasterSymmetry::resultText[base.Lookup(s)], cc.Winner(s) );
					s.PrintASCII();
					cc.UndoMove(s, n);
				}
				cc.freeMove(c);
			}
			exit(0);
		}
//		if (sym21a.Lookup(s) != sym.Lookup(s))
//		{
//			s.PrintASCII();
//			printf("State %llu has %d (old) %d (new code)\n", x, sym21a.Lookup(s), sym.Lookup(s));
//			printf("FAILURE\n");
//			exit(0);
//		}
	}
	printf("SUCCESS\n");

	
	exit(0);
}

void TestCacheSolver()
{
	CCCacheFriendlySolver::CacheSolver c(gFilePath.c_str(), true);

	// Fast ordering
	printf("---Symmetry solver without marked parents---\n");
	CCSymmetrySolver::SymmetrySolver<CCPSRank21a, false> sym21a(gFilePath.c_str(), false);

	
	CCLocalRank12 l;
	CCheckers cc;
	CCState s;
	for (int64_t x = 0; x < l.getMaxRank(); x++)
	{
		l.unrank(x, s);
		if (c.Lookup(s) != sym21a.Lookup(s))
		{
			s.PrintASCII();
			printf("State %llu has %d (new) %d (old symmetric code)\n", x, c.Lookup(s), sym21a.Lookup(s));
		}
	}

}

void MakeLocality()
{
	CCheckers cc;
	CCState s;
	CCPSRank12 r;
	int64_t max1 = r.getMaxP1Rank();
	int64_t max2 = r.getMaxP2Rank();
	int64_t p1Rank = 0;
	std::vector<bool> used(max1);
	std::vector<std::pair<int64_t, int64_t>> order;
	printf("Building order:\n");

	int64_t cnt = 0;
	for (int64_t y = 0; y < max1; y+=max1/100)
	{
		for (int64_t x = 0; x < max1; x++)
		{
			if (used[x])
				continue;

			r.unrankP1(x, s);
			// second player will now have p1Rank after next move
			cc.SymmetryFlipVertP1(s);
			s.toMove = 0; // max player
			if (r.TryAddR1(s, y)) // success
			{
				// Where in memory are we for p2?
				int64_t rank = r.rankP2(s);
				if (!used[x])
				{
					order.push_back({x, rank});
					used[x] = true;
					cnt++;
//					break;
				}
			}
		}
		printf("%lld of %lld groups assigned\n", cnt, max1);
	}
	std::sort(order.begin(), order.end(),
			  [=](const std::pair<int64_t, int64_t> &a, const std::pair<int64_t, int64_t> &b)
			  {return a.second<b.second;});
	printf("Order:\n");
	for (const auto &i : order)
	{
		printf("Group %lld at memory %lld\n", i.first, i.second);
	}
	exit(0);
}

void LocalityTest()
{
	CCheckers cc;
	CCState s;
	CCPSRank12 r;
	int64_t max1 = r.getMaxP1Rank();
	int64_t max2 = r.getMaxP2Rank();

//	int64_t cacheSize =  max2*4;
	int64_t cacheRowEntries = 2048;
	int64_t cacheRowEntriesPerGroup = 128;
	int64_t cacheRowSizeInBytes = cacheRowEntries*cacheRowEntriesPerGroup/4; // 8192 entries, 2048 bytes
	printf("%llu item cache\n", cacheRowEntries*cacheRowEntriesPerGroup*max1);
//	printf("Cache size: 2kb: 8192 entries\n");
	printf("Total cache memory usage: %1.2f GB\n", cacheRowSizeInBytes*max1/1024.0/1024.0/1024.0);
	printf("Total entries: %llu; %f in cache\n", max1*max2, cacheRowEntries*cacheRowEntriesPerGroup*1.0/max1);
	int64_t arraySize = cacheRowEntriesPerGroup*max1;
	printf("Cache array size: %llu\n", arraySize);

	std::vector<int16_t> cache(arraySize, -1);
	//int64_t usage = 0;
	int64_t hits = 0;
	int64_t load = 0;
	for (int64_t x = 0; x < max1; x++)
	{
//		int64_t minRank = max1*max2, maxRank = 0;
		for (int64_t y = max2-1; y >= 0; y--)
		{
			r.unrank(x, y, s);
			s.Reverse();
			s.toMove = 0;
			
			// Do cache test for parent
			{
				int64_t p1, p2;
				r.rank(s, p1, p2);
				//int64_t rank = r.rankP1(s);// r.rank(s);
				
				int64_t cacheLoc = p2/cacheRowEntries;
//				if (p1 > 10)
//					continue;
//				printf("[%llu] %llu - %llu - ", p1, cacheLoc, p2);
//				s.PrintASCII();
				int64_t start = p1*cacheRowEntriesPerGroup;
				bool found = false;
				for (int x = 0; x < cacheRowEntriesPerGroup; x++)
				{
					if (cache[start+x] == -1)
					{
						cache[start+x] = cacheLoc;
						load++;
						cache[start+(x+1)%cacheRowEntriesPerGroup] = -1; // clear next entry - always one free
						found = true;
						break;
					}
					else if (cache[start+x] == cacheLoc)
					{
						hits++;
						found = true;
						break;
					}
				}
//				if (cache[cacheLoc] > 0 || usage < cacheSize)
//				{
//					cache[cacheLoc]++;
//					if (cache[cacheLoc] == 1)
//						usage++;
//				}
				if (found == false)
				{
					//printf("Cache [%llu] full\n", p1);
					cache[start] = cacheLoc;
					cache[start+1] = -1;
					load++;
					//printf("%llu hits; %llu misses [%f per loaded cache]\n", hits, load, hits*1.0/load);
//					exit(0);
				}
			}
			
			
//			CCMove *m = cc.getMoves(s);
//			for (CCMove *t = m; t; t = t->next)
//			{
//				cc.ApplyMove(s, t);
//				int64_t rank = r.rank(s);
//				cc.UndoMove(s, t);
//				minRank = std::min(rank, minRank);
//				maxRank = std::max(rank, maxRank);
//			}
//			cc.freeMove(m);
		}
//		printf("\nParent: %llu. [%llu, %llu] - %llu\n", x, minRank, maxRank, maxRank-minRank);
		printf("[%llu] %llu hits; %llu misses [%f per loaded cache]\n", x, hits, load, hits*1.0/load);
//		printf("%llu used. %llu hits on %llu entries [%f] (%1.2f%% full)\n", usage, hits, cacheSize, hits*1.0/cacheSize, 100.0*(float)usage/cacheSize);
	}
}

void BigSolveForLaptop()
{
	{
		printf("---Symmetry solver with marked parents---\n");
		CCSymmetrySolver::SymmetrySolver<CCPSRank21, true> sym21a(gFilePath.c_str(), true);
	}
	{
		printf("---Symmetry solver without marked parents---\n");
		CCSymmetrySolver::SymmetrySolver<CCPSRank21, false> sym21b(gFilePath.c_str(), true);
	}
}

void TestSymmetrySolver()
{
	{
		CCPSRank12 r1;
		CCPSRank21a r2;
		CCState s;
		for (int y = 0; y < 2; y++)
		{
			if (y == 0)
				std::cout << r1.name() << "\n";
			else
				std::cout << r2.name() << "\n";
			
			for (int x = r1.getMaxRank()-100; x < r1.getMaxRank(); x++)
			{
				if (y == 0)
				{
					r1.unrank(x, s);
					s.PrintASCII();
				}
				else {
					r2.unrank(x, s);
					s.PrintASCII();
				}
			}
		}
		exit(0);
	}
	
	CCOptimizedSolver::OptimizedSolver<CCLocalRank12> o(gFilePath.c_str());
	CCSymmetrySolver::SymmetrySolver<CCPSRank12> sym12(gFilePath.c_str(), true);
	CCSymmetrySolver::SymmetrySolver<CCPSRank21> sym21(gFilePath.c_str(), true);
	CCSymmetrySolver::SymmetrySolver<CCPSRank12a> sym12a(gFilePath.c_str(), true);
	CCSymmetrySolver::SymmetrySolver<CCPSRank21a> sym21a(gFilePath.c_str(), true);

	CCLocalRank12 l;
	CCheckers cc;
	CCState s;
	for (int64_t x = 0; x < l.getMaxRank(); x++)
	{
		l.unrank(x, s);
		if (o.Lookup(s) != sym12.Lookup(s))
		{
			s.PrintASCII();
			printf("State %llu has %d (correct) %d (symmetric)\n", x, o.Lookup(s), sym12.Lookup(s));
		}
		if (o.Lookup(s) != sym21.Lookup(s))
		{
			s.PrintASCII();
			printf("State %llu has %d (correct) %d (symmetric)\n", x, o.Lookup(s), sym21.Lookup(s));
		}
		if (o.Lookup(s) != sym21a.Lookup(s))
		{
			s.PrintASCII();
			printf("State %llu has %d (correct) %d (symmetric)\n", x, o.Lookup(s), sym21a.Lookup(s));
		}
		if (o.Lookup(s) != sym12a.Lookup(s))
		{
			s.PrintASCII();
			printf("State %llu has %d (correct) %d (symmetric)\n", x, o.Lookup(s), sym12a.Lookup(s));
		}
	}
}

void CompareBitToBaseline()
{
	BackwardBitSolver::BackwardBitSolver<> bbs(gFilePath.c_str());
	CCBaselineSolver::BaselineSolver s(gFilePath.c_str(), true);
	
	s.PrintStats();
	bbs.PrintStats();
	
	if ((1))
	{
		printf("Running correctness test\n");
		uint64_t q = 0;
		CCheckers cc;
		for (uint64_t x = 0; x < cc.getMaxRank(); x++)
		{
			switch (s.Lookup(x))
			{
				case CCBaselineSolver::kIllegal:
					//printf("Illegal state %llu becomes %d\n", x, bbs.Lookup(x));
					break;
				case CCBaselineSolver::kWin:
					if (bbs.Lookup(x) != BackwardBitSolver::win)
					{
						printf("Baseline got win, forward got %s\n", BackwardBitSolver::resStr(bbs.Lookup(x)));
						q++;
					}
					break;
				case CCBaselineSolver::kLoss:
					if (bbs.Lookup(x) != BackwardBitSolver::loss)
					{
						printf("Baseline got loss, forward got %s\n", BackwardBitSolver::resStr(bbs.Lookup(x)));
						q++;
					}
					break;
				case CCBaselineSolver::kDraw:
					if (bbs.Lookup(x) != BackwardBitSolver::draw)
					{
						printf("Baseline got draw, forward got %s\n", BackwardBitSolver::resStr(bbs.Lookup(x)));
						q++;
					}
					break;
			}
		}
		if (q == 0)
		{
			printf("Test passed\n");
		}
		else {
			printf("%llu states with questionable results\n", q);
		}
	}
}

void CompareOptimizedToBaseline()
{
	CCBaselineSolver::BaselineSolver s(gFilePath.c_str(), true);
	CCOptimizedSolver::OptimizedSolver<CCDefaultRank> o(gFilePath.c_str());
	
	// force data to be built
	s.BuildData(gFilePath.c_str());
	o.BuildData(gFilePath.c_str());
	s.PrintStats();
	o.PrintStats();
	
	CCheckers cc;
	CCState state;
	for (uint64_t x = 0; x < cc.getMaxRank(); x++)
	{
		cc.unrank(x, state);
		switch (s.Lookup(state))
		{
			case CCBaselineSolver::kWin:
				if (o.Lookup(state) != CCOptimizedSolver::kWin)
					printf("mismatched win\n");
				break;
			case CCBaselineSolver::kLoss:
				if (o.Lookup(state) != CCOptimizedSolver::kLoss)
					printf("mismatched loss\n");
				break;
			case CCBaselineSolver::kDraw:
				if (o.Lookup(state) != CCOptimizedSolver::kDraw)
					printf("mismatched draw\n");
				break;
			case CCBaselineSolver::kIllegal:
				if (o.Lookup(state) != CCOptimizedSolver::kIllegal)
					printf("mismatched illegal\n");
				break;
		}
	}
}

// Compare ranking functions
void CompareRankingResults()
{
	CCOptimizedSolver::OptimizedSolver<CCLocalRank12> o2(gFilePath.c_str());
	CCOptimizedSolver::OptimizedSolver<CCLocalRank21> o3(gFilePath.c_str());
	CCOptimizedSolver::OptimizedSolver<CCDefaultRank> o1(gFilePath.c_str());
	
	CCheckers cc;
	CCState state;
	for (uint64_t x = 0; x < cc.getMaxRank(); x++)
	{
		cc.unrank(x, state);
		if (o1.Lookup(state) != o2.Lookup(state))
		{
			printf("1v2 State mismatched by ranking\n  * ");
			state.PrintASCII();
		}
		if (o1.Lookup(state) != o3.Lookup(state))
		{
			printf("1v3 State mismatched by ranking\n  * ");
			state.PrintASCII();
		}
	}
}

void VerifySymmetryCorrectness()
{
	CCOptimizedSolver::OptimizedSolver<CCLocalRank12> o2(gFilePath.c_str());
	CCheckers cc;
	CCState s1, s2;
	CCLocalRank12 r;
	
	printf("Verifying p1/p2 symmetry...\n");
	bool fail = false;
	Timer t;
	t.StartTimer();
	for (uint64_t x = 0; x < cc.getMaxRank(); x++)
	{
		r.unrank(x, s1);
		s2 = s1;
		s2.Reverse();
		
		auto res1 = o2.Lookup(s1);
		auto res2 = o2.Lookup(s2);
		if (res1 == CCOptimizedSolver::kWin && res2 != CCOptimizedSolver::kLoss)
		{
			s1.PrintASCII();
			s2.PrintASCII();
			printf("Result %d vs %d\n", o2.Lookup(s1), o2.Lookup(s2));
			fail = true;
		}
		if (res1 == CCOptimizedSolver::kDraw && res2 != CCOptimizedSolver::kDraw)
		{
			s1.PrintASCII();
			s2.PrintASCII();
			printf("Result %d vs %d\n", o2.Lookup(s1), o2.Lookup(s2));
			fail = true;
		}
		if (res1 == CCOptimizedSolver::kIllegal && res2 != CCOptimizedSolver::kIllegal)
		{
			s1.PrintASCII();
			s2.PrintASCII();
			printf("Result %d vs %d\n", o2.Lookup(s1), o2.Lookup(s2));
			fail = true;
		}
	}
	t.EndTimer();
	if (fail)
		printf("Verifiation complete; errors found. %1.2fs elapsed\n", t.GetElapsedTime());
	else
		printf("Verifiation complete; no errors. %1.2fs elapsed\n", t.GetElapsedTime());
}

#include <sys/stat.h>
bool fileExists(const char *name)
{
	struct stat buffer;
	return (stat(name, &buffer) == 0);
}

void SaveAsSVG(const CCState &s)
{
	std::string fname = "/Users/nathanst/cc/svg/cc_";
	int count = 0;
	while (fileExists((fname+std::to_string(count)+".svg").c_str()))
	{
		count++;
	}
	printf("Save to '%s'\n", (fname+std::to_string(count)+".svg").c_str());
	
	MakeSVG(s, (fname+std::to_string(count)+".svg").c_str());
}

void PrintStats()
{
	ParallelEM::Solver *emsp = new ParallelEM::Solver(gFilePath.c_str(), gFilePath.c_str(), false); // true - to clear data for build
	ParallelEM::Solver &ems = *emsp;
	ems.PrintStats();
}

void ExploreProof()
{
	ParallelEM::Solver *emsp = new ParallelEM::Solver(gFilePath.c_str(), gFilePath.c_str(), false); // true - to clear data for build
	ParallelEM::Solver &ems = *emsp;
	
	CCheckers cc;
	CCState s;
	cc.Reset(s);
	
	while (!cc.Done(s))
	{
		std::vector<CCMove *> moves;
		s.Print();
		CCMove *m = cc.getMoves(s);
		for (CCMove *n = m; n; n = n->next)
		{
			printf("[%lu]: ", moves.size());
			moves.push_back(n);
			n->Print(0);
			cc.ApplyMove(s, n);
			printf(" {%s}\n", FasterSymmetry::resultText[ems.Lookup(s)]);
			cc.UndoMove(s, n);
		}
		printf("Choice: ");
		int val;
		scanf("%d", &val);
		if (val == -2)
		{
			SaveAsSVG(s);
			for (CCMove *n = m; n; n = n->next)
			{
				cc.ApplyMove(s, n);
				SaveAsSVG(s);
				cc.UndoMove(s, n);
			}
		}
		else if (val == -1)
		{
			std::string fname = "/Users/nathanst/cc/svg/cc_";
			int count = 0;
			while (fileExists((fname+std::to_string(count)+".svg").c_str()))
			{
				count++;
			}
			printf("Save to '%s'\n", (fname+std::to_string(count)+".svg").c_str());

			MakeSVG(s, (fname+std::to_string(count)+".svg").c_str());
		}
		else
			cc.ApplyMove(s, moves[val]);
		cc.freeMove(m);
	}
	s.Print();
	printf("%d wins!\n", cc.Winner(s));
}

