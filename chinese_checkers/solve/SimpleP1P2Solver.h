//
//  SimpleP1P2Solver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 1/5/18.
//  Copyright © 2017 NS Software. All rights reserved.
//

#ifndef SimpleP1P2Solver_h
#define SimpleP1P2Solver_h

#include <cstdio>
#include <string>
#include <thread>
#include <array>
#include "CCheckers.h"
#include "NBitArray.h"
#include "SolveStats.h"
#include "CCRankings.h"
#include "TwoBitFileVector.h"
#include "SharedQueue.h"
#include "Timer.h"

namespace SimpleP1P2Solver {
	
	const int THREADS = 8;
	
	enum tResult {
		kWin = 2, kLoss = 1, kDraw = 0, kIllegal = 3
	};
		
	enum {
		kMaxPlayer = 0, kMinPlayer = 1
	};
	
	struct GroupStructure {
		uint8_t symmetryRedundant;
		uint8_t changed;
		int32_t symmetricRank;
		int32_t assignedCount;
		int32_t memoryOffset; // this is the offset of the start of the group in p2rank increments (need to multiply)
	};
	
	class Solver {
	public:
		Solver(const char *dataPath, const char *scratchPath, bool clearOldData = false);
		~Solver();
		tResult Lookup(const CCState &s) const;
		bool NeedsBuild();
		void BuildData();
		void PrintStats() const;
		void DebugState(CCState &s);
		void DebugState(int64_t r1, int64_t r2);
	private:
		void InitMetaData();
		void Initial();
		uint64_t ThreadInitial();
		
		
		void DoLoops(CCheckers &cc, int64_t max2, int64_t p1Rank);
		void ThreadMain(int64_t p1Rank);
		//void DoThreadLoop(CCheckers &cc, int thread, int64_t max2, int64_t p1Rank, CCState s, uint64_t &proven);
		void DoThreadLoop(CCheckers &cc, int64_t p1Rank, uint64_t &proven);
		
		void SinglePass();
		
		uint64_t SinglePassInnerLoop(CCheckers &cc, CCState s, int64_t r1, int64_t finalp1Rank, bool doubleFlip);
		tResult GetValue(CCheckers &cc, const CCState &s, int64_t finalp1Rank, bool doubleFlip);
		void PropagateWinToParent(CCheckers &cc, CCState &s);
		void MarkParents(CCheckers &cc, CCState &s);
		void DoBFS();
		void GetCacheOrder();
		void GetSearchOrder();
		
		void WriteToBuffer(uint32_t group, uint32_t offset, tResult value);
//		void FlushBuffers();
//		void FlushBuffer(uint32_t which);
//		void PrintBufferStats() const;
//		void FlushBiggestBuffer();
//		void ConditionalFlushBiggestBuffer();
//		void FlushBuffersInMemory();
//		void PreloadLargeBuffers(CCheckers &cc, int64_t p1Rank);
//		
		const char *GetFileName(const char *path);
		const char *GetTempFileName(int which);
		tResult Translate(const CCState &s, tResult res) const;
		tResult Translate(int nextPlayer, tResult res) const;
		// Groups per file - needs to correspond with the temporary buffer size
		static const int kNumGroups = ConstGetSymmetricP1Ranks(DIAGONAL_LEN, NUM_PIECES);
		static const int kEntriesPerGroup = ConstChoose(NUM_SPOTS-NUM_PIECES, NUM_PIECES);
		static const int kGroupsPerFile = 1<<8; // 256
		static const int kNumFiles = (kNumGroups+kGroupsPerFile-1)/kGroupsPerFile;

		NBitArray<2> p1p2data;
		NBitArray<2> p2p1data;

		uint64_t mProven;
		int64_t symmetricStates;
		stats stat;
		CCPSRank12 r;
		std::array<GroupStructure, ConstChoose(NUM_SPOTS, NUM_PIECES)> p1p2groups;
		std::array<GroupStructure, ConstChoose(NUM_SPOTS, NUM_PIECES)> p2p2groups;

		std::vector<int8_t> bfs;
		std::vector<int32_t> order;
		Timer totalTime;
	};
	
}

#endif
