//
//  InitialParallelSolver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 11/5/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#ifndef ParallelSolver_h
#define ParallelSolver_h

#include <stdio.h>

#include <stdio.h>
#include "CCheckers.h"
#include "NBitArray.h"
#include "SolveStats.h"
#include "CCRankings.h"
#include "SharedQueue.h"
#include <thread>
#include <atomic>

namespace Parallel {
	
	enum tResult {
		kWin = 2, kLoss = 1, kDraw = 0, kIllegal = 3
	};
	
	enum {
		kMaxPlayer = 0, kMinPlayer = 1
	};
	
	enum tMeta {
		kHasNoProvenChildren = 0,
		kHasProvenChildren = 1
	};
	
	struct GroupStructure {
		int32_t memoryOffset;
		int32_t symmetricRank;
	};
	
	class Solver {
	private:
		struct range {
			int64_t p1Group, from, to;
		};
		struct result {
			int64_t r1, r2;
		};
		static const int WORK_UNIT = 8;
		struct work {
			int64_t finalp1Rank;
			bool doubleFlip;
			result w[WORK_UNIT];
		};

	public:
		Solver(const char *path, bool forceBuild = false);
		~Solver();
		tResult Lookup(const CCState &s) const;
		void BuildData(const char *path, int numThreads);
		void PrintStats() const;
	private:
		void InitMetaData();
		void Initial(int numThreads);
		
		//		int64_t GetBestNextGroup();
		void SinglePass(int numThreads);

		void DoWorkThread();
		void FindWorkThread();

		void WriteResult(std::vector<result> &data, tResult valueToWrite);
		bool FindWorkInnerLoop(CCheckers &cc, int64_t p1Rank, int64_t p2Rank, CCState &s, result &r);
		tResult GetValue(CCheckers &cc, const CCState &s, int64_t finalp1Rank, bool doubleFlip);
		void PropagateWinToParent(CCheckers &cc, CCState &s, std::vector<result> &wins);
		void MarkParents(CCheckers &cc, CCState &s);
		
		void DoBFS();
		void GetSearchOrder();

		void StartThreads(int numThreads);
		void EndThreads();
		
		const char *GetFileName(const char *path);
		tResult Translate(const CCState &s, tResult res) const;
		tResult Translate(int nextPlayer, tResult res) const;
		NBitArray<2> data;
		uint64_t proven;
		int64_t symmetricStates;
		stats stat;
		CCPSRank12 r;
		GroupStructure *groups;
		std::vector<int8_t> bfs;
		std::vector<int64_t> order;
		std::atomic_flag *open;
		//std::vector<bool> open;
		
		const int WORK_SIZE = 500;
		const int CACHE_WRITE_SIZE = 1000;
		const int WORK_QUEUE_BUFFER = 200;
		uint32_t memoryMult;

		std::mutex lock;
		SharedQueue<range> findWorkQueue;
		SharedQueue<work> doWorkQueue;
		std::vector<std::thread> findWorkThreads;
		std::vector<std::thread> doWorkThreads;
};
	
}


#endif /* CacheFriendlySolver_h */
