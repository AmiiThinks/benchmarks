//
//  EMSolver.cpp
//  CC Solver
//
//  Created by Nathan Sturtevant on 11/5/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#include "ParallelEMSolver.h"
#include <cassert>
#include <cstring>
#include <thread>
#include <array>
#include <unordered_map>
#include "Timer.h"

namespace ParallelEM {
	static const char *resultText[] = {"Draw", "Loss", "Win", "Illegal"};

	const int ParallelEM::Solver::kNumGroups;
	const int ParallelEM::Solver::kEntriesPerGroup;
	const int ParallelEM::Solver::kNumFiles;

	const bool doParentPropagation = true;

#pragma mark - Initialization -

	Solver::Solver(const char *dataPath, const char *scratchPath, bool forceBuild)
	:scratchPath(scratchPath), data(dataPath, forceBuild)
	{
		if (NUM_PIECES == 2 || NUM_PIECES == 5)
		{
			printf("%d pieces not symmetric; aborting\n", NUM_PIECES);
			exit(0);
		}
		printf("### Chinese Checkers Solver! Board: %d Pieces: %d\n", NUM_SPOTS, NUM_PIECES);
		printf("--> Starting ParallelEM Solver [%s] --\n", __FILE__);
		printf("--> Ranking: %s --\n", r.name());

		
		InitMetaData();
		printf("--> Done initializing meta-data.\n");
		if (data.NeedsBuild())
			printf("--> Waiting to build solve data\n");
		else
			printf("--> Build data ready for queries.\n");
	}
	
	Solver::~Solver()
	{
		//delete [] threadCC;
	}

	void Solver::InitMetaData()
	{
		CCheckers cc;
		CCState s;
		int32_t memoryOffset = 0;
		
		printf("--> Initializing data and detecting symmetry\n");
		//uint64_t maxRank = r.getMaxRank();
		assert(groups.size() == r.getMaxP1Rank());
		//groups.resize(r.getMaxP1Rank());
		
		// Note that in single-player CC we have to (maybe?) generate
		// the children of both the regular and flipped states
		
		int64_t totalStates = 0;
		statesOnDisk = maxStatesOnDisk = statesWrittenToDisk = 0;
		symmetricStates = 0;
		for (int32_t x = 0; x < groups.size(); x++)
			groups[x].symmetricRank = -1;
		for (int32_t x = 0; x < groups.size(); x++)
		{
			groups[x].changed = 0;
			groups[x].assignedCount = 0;
			r.unrank(x, 0, s);
			CCState sym;
			cc.FlipPlayer(s, sym, 0);
			
			// Since we are only going up to 7x7 (6 piece) CC, we can narrow to 32 bits here
			int32_t otherRank = static_cast<int32_t>(r.rankP1(sym));
			if (otherRank < x)
			{
				groups[x].symmetryRedundant = true;
				groups[x].symmetricRank = otherRank;
				groups[otherRank].symmetricRank = x;
			}
			else {
				groups[x].memoryOffset = memoryOffset;
				memoryOffset++;//= r.getMaxP2Rank();
				groups[x].symmetryRedundant = false;
				//printf("[%d] !Symmetric: ", x); s.PrintASCII();
				symmetricStates += r.getMaxP2Rank();
			}
			totalStates += r.getMaxP2Rank();
		}
		
		// 6,482 ? (non symmetric groups?)
		//		data.Resize(symmetricStates);
		//		data.Clear();
		//printf("--> Test: (49 3) = %llu\n", ConstChoose(49, 3));
		printf("--> SYMMETRY: %lld total states; %lld after symmetry reduction\n", totalStates, symmetricStates);
		printf("--> STATIC: %llu fully symmetric groups\n", ConstGetNonMiddleSymmetry(DIAGONAL_LEN, NUM_PIECES, 0));
		printf("--> STATIC SYMMETRY: %llu non-symmetric groups; %lld total states\n",
			   ConstGetSymmetricP1Ranks(DIAGONAL_LEN, NUM_PIECES), (uint64_t)ConstChoose(NUM_SPOTS-NUM_PIECES, NUM_PIECES)*(uint64_t)ConstGetSymmetricP1Ranks(DIAGONAL_LEN, NUM_PIECES));
		printf("--> STATIC FINAL: %llu groups; %lld states per group\n",
			   (uint64_t)ConstGetSymmetricP1Ranks(DIAGONAL_LEN, NUM_PIECES),
			   (uint64_t)ConstChoose(NUM_SPOTS-NUM_PIECES, NUM_PIECES));
		printf("--> Running computation to order search\n");
		GetSearchOrder();
		
		memoryOffset = 0;
		printf("--> Constructing ordering\n");
		for (int32_t x = 0; x < order.size(); x++)
		{
			if (!groups[order[x]].symmetryRedundant)
			{
				//printf(" [%d] Group %d is at memory %llu\n", x, order[x], memoryOffset);
				groups[order[x]].memoryOffset = memoryOffset;
				memoryOffset += 1;//r.getMaxP2Rank();
			}
		}
		printf("--> %lu reduced groups [from %lu - %1.3fx]\n", order.size(), groups.size(), groups.size()/(float)order.size());
		printf("--> Disk buffer entry size: %lu [should be 4 if bit fields are implemented as expected]\n", sizeof(DiskEntry));
		assert(1+(memoryOffset>>8) == kNumFiles);
		//diskBuffer.resize(1+(memoryOffset>>8));
		for (int x = 0; x < diskBuffer.size(); x++)
		{
			diskBuffer[x].f = fopen(GetTempFileName(x), "w+b"); // open and truncate file
			assert(diskBuffer[x].f != 0);
			//setvbuf(diskBuffer[x].f, NULL, _IONBF, 0);
			for (int y = 0; y < THREADS; y++)
			{
				diskBuffer[x].threadData[y].index = 0;
				diskBuffer[x].threadData[y].onDisk = 0;
			}
		}
		
		//int numFiles = (kNumGroups+kGroupsPerFile-1)/kGroupsPerFile;
		int filesPerThread = (kNumFiles+THREADS-1)/THREADS;
//		groupsPerThread = filesPerThread*kGroupsPerFile;
		printf("--> Using %d threads on %d files\n", THREADS, kNumFiles);
//		printf("--> Assigning %d files and %d groups per thread\n", filesPerThread, groupsPerThread);
//		printf("--> Last thread gets %d groups\n", groupsPerThread-(groupsPerThread*THREADS-kNumGroups));
		//threadCC = new CCheckers[THREADS];
	}
	
	void Solver::GetCacheOrder()
	{
		CCheckers cc;
		CCState s;
		int64_t max1 = r.getMaxP1Rank();
		std::vector<bool> used(max1);
		std::vector<std::pair<int64_t, int64_t>> cacheOrder;
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
						cacheOrder.push_back({x, rank});
						used[x] = true;
						cnt++;
						//					break;
					}
				}
			}
			if (cnt == max1)
				break;
			//printf("%lld of %lld groups assigned\n", cnt, max1);
		}
		std::sort(cacheOrder.begin(), cacheOrder.end(),
				  [=](const std::pair<int64_t, int64_t> &a, const std::pair<int64_t, int64_t> &b)
				  {return a.second<b.second;});
		printf("Order:\n");
		for (const auto &i : cacheOrder)
		{
			//printf("Group %lld at memory %lld\n", i.first, i.second);
			if (!groups[i.first].symmetryRedundant)
				order.push_back(i.first);
		}
	}
	
	void Solver::GetSearchOrder()
	{
//		printf("--> ORDER: Cache optimized\n");
//		GetCacheOrder();
//		return;
		
//		//for (int x = groups.size()-1; x >= 0; x--)
//		for (int x = 0; x < groups.size(); x++)
//			if (!groups[x].symmetryRedundant)
//				order.push_back(x);
//		std::reverse(order.begin(), order.end());
//		printf("--> ORDER: default (reversed)\n");
//		return;
		
		DoBFS();
		printf("--> ORDER: BFS (reversed)\n");

		int newItems;
		int d = 0;
		order.clear();
		do {
			newItems = 0;
			//printf("Search order for %d\n", d);
			for (int32_t x = bfs.size()-1; x >= 0; x--)
			{
				if (bfs[x] == d) // No redundant states in the order now
				{
					if (!groups[x].symmetryRedundant)
					{
						order.push_back(x);
						newItems++;
						//if (d == 0) printf("%d not redundant\n", x);
					} else {
						//if (d == 0) printf("%d redundant\n", x);
					}
				}
			}
			d++;
		} while (newItems > 0);
	}
	
	void Solver::DoBFS()
	{
		CCheckers cc;
		CCState s;
		Timer t;
		bfs.resize(r.getMaxP1Rank());
		std::fill(bfs.begin(), bfs.end(), -1);
		
		//		cc.Reset(s);
		//		printf("Reset: "); s.PrintASCII();
		//		cc. SymmetryFlipHoriz(s);
		//s.Reverse();
		// Set the start state to 0
		
		//		bfs[r.rankP1(s)] = 0;
		cc.ResetP1Goal(s);
		bfs[r.rankP1(s)] = 0;
		//r.unrankP1(r.getMaxP1Rank()-1, s);
		//printf("Start: "); s.PrintASCII();
		
		int depth = 0;
		int written = 0;
		int total = 1;
		printf("--> BFS: ");
		t.StartTimer();
		do {
			written = 0;
			for (int x = 0; x < r.getMaxP1Rank(); x++)
			{
				if (bfs[x] == depth)
				{
					r.unrankP1(x, s);
					CCMove *m = cc.getMoves(s);
					for (CCMove *tmp = m; tmp; tmp = tmp->next)
					{
						cc.ApplyMove(s, tmp);
						s.toMove = kMaxPlayer;
						int64_t rank = r.rankP1(s);
						if (bfs[rank] == -1)
						{
							bfs[rank] = depth+1;
							written++;
						}
						s.toMove = kMinPlayer;
						cc.UndoMove(s, tmp);
					}
					cc.freeMove(m);
				}
			}
			total += written;
			printf("%d ", depth);
			//printf("Depth %d complete. %d new. %d of %d complete\n", depth, written, total, r.getMaxP1Rank());
			depth++;
		} while (written != 0);
		t.EndTimer();
		printf("\n--> BFS complete to depth %d in %1.2fs\n", depth, t.GetElapsedTime());
	}
	
#pragma mark - mark initial wins/losses -
	
	void Solver::ThreadInitial(int whichThread)
	{
		// TODO: Current status: verify whether we need to flush buffers below. Then test.
		CCheckers cc;
		CCState s, tmp;
		int64_t max2 = r.getMaxP2Rank();
		int64_t localProven = 0;

		while (true)
		{
			int item;
			workUnits.WaitRemove(item);
			if (item == -1)
				break;
			
			for (int64_t x = 0; x < order.size(); x++)
			{
				int64_t p1Rank = order[x];
				
				if (groups[p1Rank].symmetryRedundant)
					continue;

				// Only do work in this buffer
				if (item != groups[p1Rank].memoryOffset>>8)
					continue;
				
				r.unrankP1(p1Rank, s);
				
				int startCount = cc.GetNumPiecesInStart(s, 0);
				int goalCount = cc.GetNumPiecesInGoal(s, 0);
				// 1. if no pieces in home, then only one possible goal.
				if (startCount == 0 && goalCount == 0)
				{
					bool result = cc.SetP2Goal(s);
					assert(result == true);
					//r.unrankP2(0, s);
					//stat.unrank++;
					
					if (cc.Winner(s) == -1)
						continue;
					//if (data.SetIf(groups[p1Rank].memoryOffset*r.getMaxP2Rank(), kLoss, kDraw))
					if (data.SetIf(groups[p1Rank].memoryOffset, 0, kLoss, kDraw))
					{
						groups[p1Rank].assignedCount++;
						groups[p1Rank].changed = true;
						// TODO: When this is parallelized, needs to be updated
						localProven++;
						PropagateWinToParent(cc, s, whichThread); // This is a win at the parent!
					}
					continue;
				}
				
				// 2. if pieces in home, then try all goals (could do better)
				for (int64_t p2Rank = 0; p2Rank < max2; p2Rank++)
				{
					if (p2Rank > 0)
					{
						// clear p2 pieces
						for (int x = 0; x < NUM_PIECES; x++)
						{
							s.board[s.pieces[1][x]] = 0;
						}
					}
					r.unrankP2(p2Rank, s);
					stat.unrank++;
					
					if (!cc.Legal(s))
					{
						auto v = data.Get(groups[p1Rank].memoryOffset, p2Rank);
						if (v == kDraw)
						{
							// TODO: When this is parallelized, needs to be updated
							localProven++;
							groups[p1Rank].assignedCount++;
							groups[p1Rank].changed = true;
						}
						data.Set(groups[p1Rank].memoryOffset, p2Rank, kIllegal);
						//data.Set(groups[p1Rank].memoryOffset*r.getMaxP2Rank()+p2Rank, kIllegal);
						continue;
					}
					
					switch (cc.Winner(s))
					{
						case -1: // no winner
							break;
						case 0: // not possible, because it's always player 0's turn
							// Actually, this is possible in one situation - but we consider
							// it illegal to make a suicide move to lose the game
							assert(!"(0) This isn't possible");
							break;
						case 1:
							//if (data.SetIf(groups[p1Rank].memoryOffset*r.getMaxP2Rank()+p2Rank, kLoss, kDraw))
							if (data.SetIf(groups[p1Rank].memoryOffset, p2Rank, kLoss, kDraw))
							{
								// TODO: When this is parallelized, needs to be updated
								localProven++;
								groups[p1Rank].assignedCount++;
								groups[p1Rank].changed = true;
								tmp = s;
								PropagateWinToParent(cc, tmp, whichThread); // This is a win at the parent!
							}
							else if (data.SetIf(groups[p1Rank].memoryOffset, p2Rank, kLoss, kWin))
							{
								//proven++;
								//groups[p1Rank].assignedCount++;
								groups[p1Rank].changed = true;
								tmp = s;
								PropagateWinToParent(cc, tmp, whichThread); // This is a win at the parent!
							}
							break;
					}
				}
			}
		}
		work[whichThread].WaitAdd(localProven);
		return;
	}
//	SharedQueue<int> workUnits;

	void Solver::Initial()
	{
		for (int thread = 0; thread < THREADS; thread++)
		{
			threads[thread] = std::thread(&Solver::ThreadInitial, this, thread);
		}
		for (int x = 0; x < kNumFiles; x++)
		{
			workUnits.WaitAdd(x);
		}
		for (int thread = 0; thread < THREADS; thread++)
			workUnits.WaitAdd(-1);

		for (int thread = 0; thread < THREADS; thread++)
		{
			int64_t count;
			work[thread].WaitRemove(count);
			mProven += count;
		}
		for (int thread = 0; thread < THREADS; thread++)
		{
			threads[thread].join();
		}
	}
//	{
//		CCheckers cc;
//		CCState s, tmp;
//		//		uint64_t maxRank = r.getMaxRank();
//		//		int64_t max1 = r.getMaxP1Rank();
//		int64_t max2 = r.getMaxP2Rank();
//
//		for (int64_t x = 0; x < order.size(); x++)
//		{
//			int64_t p1Rank = order[x];
//
//			if (groups[p1Rank].symmetryRedundant)
//				continue;
//
//			FlushBuffer(groups[p1Rank].memoryOffset>>8);
//
//			r.unrankP1(p1Rank, s);
//
//			int startCount = cc.GetNumPiecesInStart(s, 0);
//			int goalCount = cc.GetNumPiecesInGoal(s, 0);
//			// 1. if no pieces in home, then only one possible goal.
//			if (startCount == 0 && goalCount == 0)
//			{
//				bool result = cc.SetP2Goal(s);
//				assert(result == true);
//				//r.unrankP2(0, s);
//				//stat.unrank++;
//
//				if (cc.Winner(s) == -1)
//					continue;
//				//if (data.SetIf(groups[p1Rank].memoryOffset*r.getMaxP2Rank(), kLoss, kDraw))
//				if (data.SetIf(groups[p1Rank].memoryOffset, 0, kLoss, kDraw))
//				{
//					groups[p1Rank].assignedCount++;
//					groups[p1Rank].changed = true;
//					// TODO: When this is parallelized, needs to be updated
//					mProven++;
//					PropagateWinToParent(cc, s, 0); // This is a win at the parent!
//				}
//				continue;
//			}
//
//			// 2. if pieces in home, then try all goals (could do better)
//			for (int64_t p2Rank = 0; p2Rank < max2; p2Rank++)
//			{
//				if (p2Rank > 0)
//				{
//					// clear p2 pieces
//					for (int x = 0; x < NUM_PIECES; x++)
//					{
//						s.board[s.pieces[1][x]] = 0;
//					}
//				}
//				r.unrankP2(p2Rank, s);
//				stat.unrank++;
//
//				if (!cc.Legal(s))
//				{
//					auto v = data.Get(groups[p1Rank].memoryOffset, p2Rank);
//					if (v == kDraw)
//					{
//						// TODO: When this is parallelized, needs to be updated
//						mProven++;
//						groups[p1Rank].assignedCount++;
//						groups[p1Rank].changed = true;
//					}
//					data.Set(groups[p1Rank].memoryOffset, p2Rank, kIllegal);
//					//data.Set(groups[p1Rank].memoryOffset*r.getMaxP2Rank()+p2Rank, kIllegal);
//					continue;
//				}
//
//				switch (cc.Winner(s))
//				{
//					case -1: // no winner
//						break;
//					case 0: // not possible, because it's always player 0's turn
//						// Actually, this is possible in one situation - but we consider
//						// it illegal to make a suicide move to lose the game
//						assert(!"(0) This isn't possible");
//						break;
//					case 1:
//						//if (data.SetIf(groups[p1Rank].memoryOffset*r.getMaxP2Rank()+p2Rank, kLoss, kDraw))
//						if (data.SetIf(groups[p1Rank].memoryOffset, p2Rank, kLoss, kDraw))
//						{
//							// TODO: When this is parallelized, needs to be updated
//							mProven++;
//							groups[p1Rank].assignedCount++;
//							groups[p1Rank].changed = true;
//							tmp = s;
//							PropagateWinToParent(cc, tmp, 0); // This is a win at the parent!
//						}
//						else if (data.SetIf(groups[p1Rank].memoryOffset, p2Rank, kLoss, kWin))
//						{
//							//proven++;
//							//groups[p1Rank].assignedCount++;
//							groups[p1Rank].changed = true;
//							tmp = s;
//							PropagateWinToParent(cc, tmp, 0); // This is a win at the parent!
//						}
//						break;
//				}
//			}
//		}
//
//	}
	
#pragma mark - main solver -
	
	/*
	 * Note that s will be modified in this function
	 */
	void Solver::PropagateWinToParent(CCheckers &cc, CCState &s, int thread)
	{
		if (!doParentPropagation) return;
		
		CCState tmp;
		// Always called from Loss at max node
		assert(s.toMove == kMaxPlayer);
		stat.backwardExpansions++;
		
		// Flip the whole board then generate all moves. More efficient than
		// Generating moves and then flipping each time.
		cc.SymmetryFlipVert(s);
		assert(s.toMove == kMinPlayer);
		CCMove *m = cc.getReverseMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			int64_t p1Rank, p2Rank;
			
			cc.ApplyReverseMove(s, t);
			
			p1Rank = r.rankP1(s);
			
			stat.rank++;
			if (!groups[p1Rank].symmetryRedundant)
			{
				p2Rank = r.rankP2(s);
				WriteToBuffer(p1Rank, p2Rank, kWin, thread);
			}
			
			// Only flip pieces
			cc.SymmetryFlipHoriz_PO(s, tmp);
			
			p1Rank = r.rankP1(tmp);
			
			stat.rank++;
			if (!groups[p1Rank].symmetryRedundant)
			{
				p2Rank = r.rankP2(tmp);
				WriteToBuffer(p1Rank, p2Rank, kWin, thread);
			}
			cc.UndoReverseMove(s, t);
		}
		cc.freeMove(m);
	}
	
	
	/*
	 * Returns the value for the parent, so it has to be flipped.
	 */
	tResult Solver::GetValue(CCheckers &cc, const CCState &s, int64_t finalp1Rank, bool doubleFlip, uint8_t *readBuffer)
	{
		CCState tmp;
		// Max player to move
		if (doubleFlip)
			cc.SymmetryFlipHorizVert(s, tmp); // Same flip used initially to flip players - returns to initial p1rank
		else
			cc.SymmetryFlipVert(s, tmp); // Same flip used initially to flip players - returns to initial p1rank
		assert(tmp.toMove == kMinPlayer);
		
		CCMove *m = cc.getMoves(tmp);
		stat.forwardExpansions++;
		tResult result = kLoss;
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyMove(tmp, t); // Now max player to move
			int64_t p1Rank, p2Rank;
			tResult val;
			// We re-use the initial p1Rank and just compute p2 here
			assert(tmp.toMove == kMaxPlayer);
			p1Rank = finalp1Rank;
			p2Rank = r.rankP2(tmp);
			tResult tmpVal = (tResult)data.GetReadOnlyBuffer(groups[p1Rank].memoryOffset, p2Rank, readBuffer);
			val = Translate(kMaxPlayer, tmpVal); // Invert back to get value for min player
			stat.rank++;
			cc.UndoMove(tmp, t);
			
			
			switch (val)
			{
				case kWin:
					if (tmp.toMove == kMaxPlayer)
					{
						assert(!"(a) Shouldn't get here - it's min's turn\n");
						cc.freeMove(m);
						return kIllegal;
					}
					break;
				case kLoss: // loss at child will be win for parent
					if (tmp.toMove == kMinPlayer)
					{
						cc.freeMove(m);
						//						assert(!"(b) Shouldn't get here - a loss to min should have been propagated immediately(?).\n");
						// Previously we didn't get here because wins are propagated immediately to parents
						// But, when we send such states to disk, they are missed in memory and may be found later,
						// so we need to handle them here.
						return kWin;
					}
					break;
				case kDraw: // still unknown // NOTE: By immediately propagating wins we can skip out once we find a draw.
					if (doParentPropagation)
					{
						cc.freeMove(m);
						return kDraw;
					}
					else {
						result = kDraw;
					}
					break;
				case kIllegal: // ignore
					break;
			}
			
		}
		cc.freeMove(m);
		// Parent (max node) is a loss only if all children are losses (wins for player to move)
		return result;
	}
	
	uint64_t Solver::SinglePassInnerLoop(CCheckers &cc, CCState s, int64_t r1, int64_t finalp1Rank, bool doubleFlip, int thread, uint8_t *readBuffer)
	{
		uint64_t proven = 0;
		assert(s.toMove == kMaxPlayer);
		if (groups[r1].symmetryRedundant)
		{
			// We can look up the children, but we can't store the parent value. Skip.
			return proven;
		}
		int64_t r2 = r.rankP2(s);

//		static int64_t hit = 0, miss = 0;
//		if (data.IsInCache(groups[r1].memoryOffset, r2))
//			hit++;
//		else miss++;
//		if (0 == (hit+miss)%1000000)
//			printf("%lld hit; %lld miss - %1.2f%% hit cache\n", hit, miss, hit/(hit+miss+0.0));

		// Check whether we need to prove this state
		if (data.Get(groups[r1].memoryOffset, r2) != kDraw)
			return proven;
		
		tResult result = GetValue(cc, s, finalp1Rank, doubleFlip, readBuffer);
		
		switch (result)
		{
			case kWin:
				// Normally we don't see wins at max nodes, since they are immediately propagated. But, with delayed
				// processing of parents we might miss some.
				//if (data.SetIf(groups[r1].memoryOffset*r.getMaxP2Rank()+r2, kWin, kDraw))
				if (data.SetIf(groups[r1].memoryOffset, r2, kWin, kDraw))
				{
					groups[r1].lock.lock();
					groups[r1].assignedCount++;
					groups[r1].changed = true;
					groups[r1].lock.unlock();
					proven++;
				}
				break;
			case kLoss:
				// Loss symmetrically becomes win at parent
				//if (data.SetIf(groups[r1].memoryOffset*r.getMaxP2Rank()+r2, kLoss, kDraw))
				if (data.SetIf(groups[r1].memoryOffset, r2, kLoss, kDraw))
				{
					PropagateWinToParent(cc, s, thread);
					groups[r1].lock.lock();
					groups[r1].assignedCount++;
					groups[r1].changed = true;
					groups[r1].lock.unlock();
					proven++;
				}
				break;
			case kDraw: // still unknown
				break;
			case kIllegal:
				assert(!"(d) Shouldn't get here");
				break;
		}
		return proven;
	}
	
	void Solver::ThreadMain(int whichThread)
	{
		uint8_t readBuffer[data.GetReadBufferSize()];
		CCheckers cc;
		while (true)
		{
			// 1. get work - provides the p1Rank being done this round
			int64_t p1Rank;
			uint64_t proven;
			uint64_t totalProven = 0;
			work[whichThread].WaitRemove(p1Rank);
			if (p1Rank == -1)
				return;
			data.CopyReadOnlyBuffer(readBuffer);
			// 2. get individual work units
			int whichFile;
			while (true) {
				workUnits.WaitRemove(whichFile);
				if (whichFile == -1)
					break;
				DoThreadLoop(cc, whichThread, whichFile, p1Rank, proven, readBuffer);
				totalProven += proven;
			}
			results[whichThread].WaitAdd(totalProven);
		}
	}
//#include <sys/kdebug_signpost.h>

	void Solver::DoThreadLoop(CCheckers &cc, int whichThread, int whichFile, int64_t p1Rank, uint64_t &proven, uint8_t *readBuffer)
	{
		CCState s;
		proven = 0;
		int64_t firstRank = whichFile*kGroupsPerFile;
		int64_t lastRank = std::min((whichFile+1)*kGroupsPerFile, kNumGroups);
//		printf("-->Thread %d doing groups from %lld to %lld\n", thread, firstRank, lastRank);
		int64_t r1;//, r2;
//		kdebug_signpost_start(3, 0, 0, 0, 2);
		
		// There are max2 possible ranks, but these are interleaved depending on the p2 pieces
		// So, we need to be able to jump to the correct rank
		for (int64_t x = firstRank; x < lastRank; x++)
		{
			if (groups[order[x]].assignedCount == kEntriesPerGroup)
				continue;
			// Start with p1Rank and any p2 state
//			r.unrank(p1Rank, max2-1, s);
			// Start with symmetric rank and no p2 state
			r.unrankP1(p1Rank, s);
			// second player will now have p1Rank after next move
			cc.SymmetryFlipVertP1(s);
			s.toMove = kMaxPlayer;
			if (r.TryAddR1(s, order[x])) // success
			{
//				s.PrintASCII();
//				r1 = r.rankP1(s);
//				assert(r1 == order[x]);
				r1 = order[x];
				proven += SinglePassInnerLoop(cc, s, r1, p1Rank, false, whichThread, readBuffer);
			}
			
			if (groups[p1Rank].symmetricRank != -1) // There is a symmetric group that leads here
			{
				// Start with symmetric rank and any p2 state
				//r.unrank(groups[p1Rank].symmetricRank, max2-1, s);
				// Start with symmetric rank and no p2 state
				r.unrankP1(groups[p1Rank].symmetricRank, s);
				// second player will now have symmetric->p1Rank after next move
				cc.SymmetryFlipVertP1(s);
				s.toMove = kMaxPlayer;

				if (r.TryAddR1(s, order[x])) // success
				{
					r1 = order[x];
//					r1 = r.rankP1(s);
//					assert(r1 == order[x]);
					proven += SinglePassInnerLoop(cc, s, r1, p1Rank, true, whichThread, readBuffer);
				}
			}

		}
//		kdebug_signpost_end(3, 0, 0, 0, 2);

	}

	void Solver::DoLoops(CCheckers &cc, int64_t max2, int64_t p1Rank)
	{
		// Flush any buffers that are in memory
		if (doParentPropagation)
		{
//			kdebug_signpost_start(1, 0, 0, 0, 0);
			FlushBuffersInMemory();

			// 1. Load (if needed) the chunk [p1Rank] we are currently reading from
			// 2. Pull any cached states from disk into memory

			// temp disk usage is too big
			//
			if ((statesOnDisk*4) > (symmetricStates/4)*1.5)
			{
				auto i = statesOnDisk;
				int cnt = 0;
				while ((statesOnDisk*4) > (symmetricStates/4)*0.5)
				{
					cnt++;
					FlushBiggestBuffer(); // TODO: Print which buffers are being flushed. Maybe compute duplicates?
				}
				printf("[f%llu-%d]\n", i-statesOnDisk, cnt);
			}
//			else {//if ((statesOnDisk*4) > (symmetricStates/4)*0.25)
//				ConditionalFlushBiggestBuffer();
//			}
//			kdebug_signpost_end(1, 0, 0, 0, 0);
		}

		FlushBuffer(groups[p1Rank].memoryOffset>>8);

		// [??] Read is only valid until we flush buffers
		data.LoadLargeBufferForWrite(groups[p1Rank].memoryOffset>>8);
		data.LoadReadOnlyBuffer(groups[p1Rank].memoryOffset);
		
//		kdebug_signpost_start(2, 0, 0, 0, 1);
		// Tell all the threads we are doing this p1Rank
		for (int thread = 0; thread < THREADS; thread++)
			work[thread].Add(p1Rank);
		// send each of the work units across
		for (int x = 0; x < kNumFiles; x++)
			workUnits.WaitAdd(x);
		for (int thread = 0; thread < THREADS; thread++)
			workUnits.WaitAdd(-1);
		for (int thread = 0; thread < THREADS; thread++)
		{
			uint64_t items;
			results[thread].WaitRemove(items);
			mProven += items;
		}
//		kdebug_signpost_end(2, 0, 0, 0, 1);
	}

	void Solver::SinglePass()
	{
		CCheckers cc;
		int64_t max2 = r.getMaxP2Rank();
		float perc = 0.05f;
		
		for (int64_t x = 0; x < order.size(); x++)
		{
			int64_t p1Rank = order[x];
			if ((float)x/order.size() > perc)
			{
				printf("%1.1f%% [%1.2fs]", x*100.0/order.size(), totalTime.GetElapsedTime());
				perc += 0.05;
				if (doParentPropagation)
					PrintBufferStats();
			}
			if (!groups[p1Rank].changed)
			{
				// If nothing in this group changed, then the parent proof can't be
				// changed by re-analyzing these states
				continue;
			}
			if (groups[p1Rank].symmetryRedundant)
			{
				// No states in this group to analyze
				continue;
			}
			groups[p1Rank].changed = false;
			
			// Loop over all parents with children in the p1Rank
			DoLoops(cc, max2, p1Rank);
		}
		printf("\n");
	}
	
	bool Solver::NeedsBuild()
	{
		return data.NeedsBuild();
	}
	
	// Full size:
	// --> SYMMETRY: 85251690988464 total; 42645604101646 symmetric
	void Solver::BuildData()
	{
		Timer t;
		CCheckers cc;
		//		printf("--> Initializing Meta Data\n");
		//		InitMetaData();
		mProven = 0;
		
		totalTime.StartTimer();
		t.StartTimer();
		printf("** Filling in initial states\n");
		Initial();
		t.EndTimer();
		printf("Round 0; %llu new; %llu of %llu proven; %1.2fs elapsed; %llu on disk (%llu max)\n", mProven, mProven, symmetricStates, t.GetElapsedTime(), statesOnDisk, maxStatesOnDisk);
		
		printf("** Starting up %d threads\n", THREADS);
		for (int thread = 0; thread < THREADS; thread++)
		{
			threads[thread] = std::thread(&Solver::ThreadMain, this, thread);
		}

		uint64_t oldProven;
		
		CCState start;
		cc.Reset(start);
		int64_t startR1, startR2;
		bool startProven = false;
		r.rank(start, startR1, startR2);
		printf("** Starting Main Loop\n");
		int iteration = 0;
		do {
			iteration++;
			t.StartTimer();
			oldProven = mProven;
			SinglePass();
			t.EndTimer();
			printf("Round %d; %llu new; %llu of %llu proven; %1.2fs elapsed; %llu on disk (%llu max)\n", iteration, mProven-oldProven, mProven, symmetricStates, t.GetElapsedTime(), statesOnDisk, maxStatesOnDisk);
			
			if (!startProven && data.Get(groups[startR1].memoryOffset, startR2) != kDraw)
			{
				printf("Start state proven to be %s\n", resultText[data.Get(groups[startR1].memoryOffset, startR2)]);
				startProven = true;
			}
			// Make sure we don't have unproven states waiting to be flushed
			if (mProven == oldProven)
				FlushBuffers();
		} while (mProven != oldProven);
		totalTime.EndTimer();
		printf("%1.2fs total time elapsed\n", totalTime.GetElapsedTime());
		printf("%lld states written during proof\n", statesWrittenToDisk);
		//if (!startProven)
		printf("Start state is a %s\n", resultText[data.Get(groups[startR1].memoryOffset, startR2)]);

		// TODO: Restore this
		//data.Write(GetFileName());
		PrintStats();
	}
	
#pragma mark - disk buffer code -
	
	void Solver::WriteToBuffer(uint32_t group, uint32_t offset, tResult value, int thread)
	{
		if (groups[group].assignedCount == kEntriesPerGroup)
		{
			//printf("Group %d has %llu assigned of %llu total\n", group, groups[group].assignedCount, kEntriesPerGroup);
			return;
		}

//		// 0. If we can write to the large buffer, do so immediately.
//		// [Note: Removed because writing one state is inefficient - this write requires a lock]
//		bool success;
//		bool result = data.SetIfInLargeBuffer(groups[group].memoryOffset, offset, kWin, kDraw, success);
//		if (success)
//		{
//			if (result)
//			{
//				mProven++;
//				groups[group].lock.lock();
//				groups[group].assignedCount++;
//				groups[group].changed = true;
//				groups[group].lock.unlock();
//			}
//			return;
//		}
		
//		groups[group].lock.lock();
//		groups[group].onDisk++;
//		groups[group].lock.unlock();
		
		// 2. Otherwise, buffer and write to disk until later.
		//		assert(value == kWin);
		// offset needs 43!/37!6! entries which requires 23 bits
		// Let's use 24 bits for offset and 8 bits for collapsing groups
		auto &i = diskBuffer[groups[group].memoryOffset>>8];
		if (i.threadData[thread].index >= queueSize) {
			fwrite(i.threadData[thread].queue, sizeof(DiskEntry), queueSize, i.f);
			i.threadData[thread].index = 0;
			i.threadData[thread].onDisk++;
 
			diskCountLock.lock();
			statesOnDisk+=queueSize;
			maxStatesOnDisk = std::max(statesOnDisk, maxStatesOnDisk);
			statesWrittenToDisk+=queueSize;
			diskCountLock.unlock();
		}
		i.threadData[thread].queue[i.threadData[thread].index] = { static_cast<uint32_t>(groups[group].memoryOffset&0xFF), offset };
		i.threadData[thread].index++;
	}
	
	
	void Solver::PrintBufferStats() const
	{
		int max_b = 0, min_b = 0xFFFFFFF;
		for (int x = 0; x < diskBuffer.size(); x++)
		{
			for (int y = 0; y < THREADS; y++)
			{
				max_b = std::max(max_b, diskBuffer[x].threadData[y].onDisk*queueSize);
				min_b = std::min(min_b, diskBuffer[x].threadData[y].onDisk*queueSize);
			}
		}
		
		printf(" - Total states on disk: %llu; avg per bucket: %f; max: %d; min: %d\n", statesOnDisk, diskBuffer.size()/(float)statesOnDisk,
			   max_b, min_b);
	}
	
	void Solver::PreloadLargeBuffers(CCheckers &cc, int64_t p1Rank)
	{
		int loadedCount = 0;
		CCState s;
		cc.unrankPlayer(p1Rank, s, 0);
		CCMove *m = cc.getMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyMove(s, t);

			int64_t rank = cc.rankPlayer(s, 0);
			if (data.LoadLargeBufferForWrite(groups[rank].memoryOffset>>8))
				loadedCount++;
			cc.UndoMove(s, t);
		}
		cc.freeMove(m);
//		if (loadedCount > 0)
//			printf("%d new buffers preloaded\n", loadedCount);
	}
	
	// This is only called outside threading
	void Solver::FlushBuffersInMemory()
	{
		for (int x = 0; x < diskBuffer.size(); x++)
		{
			if (data.IsFileInMemory(x))
			{
				FlushBuffer(x);
			}
		}
	}
	
	// This is only called outside threading
	void Solver::FlushBiggestBuffer()
	{
		int max_ = -1;
		uint64_t maxCount = 0;
		for (int x = 0; x < diskBuffer.size(); x++)
		{
			uint64_t onDisk1 = 0;
			for (int t = 0; t < THREADS; t++)
			{
				onDisk1 += diskBuffer[x].threadData[t].onDisk;
			}
			//if (diskBuffer[x].onDisk > diskBuffer[max_].onDisk)
			if (onDisk1 > maxCount)
			{
				max_ = x;
				maxCount = onDisk1;
			}
		}
		//printf(" (file %d) ", max_);
		if (max_ != -1)
			FlushBuffer(max_);
	}
	
	// This is only called outside threading
	void Solver::ConditionalFlushBiggestBuffer()
	{
		int max_ = -1;
		uint64_t maxCount = 0;
		for (int x = 0; x < diskBuffer.size(); x++)
		{
			uint64_t onDisk1 = 0;
			for (int t = 0; t < THREADS; t++)
			{
				onDisk1 += diskBuffer[x].threadData[t].onDisk*queueSize;
			}
			//if (diskBuffer[x].onDisk > diskBuffer[max_].onDisk)
			if (onDisk1 > maxCount)
			{
				max_ = x;
				maxCount = onDisk1;
			}
		}
		// Each state is 4 bytes on backing store on disk, 2 bits in actual data
		// If the backing store is larger than [Nx - twice] the actual data, we can flush
		if (maxCount*4 > 2*kGroupsPerFile*kEntriesPerGroup/4)
		{
			printf("cf%lld\n", maxCount);
			FlushBuffer(max_);
		}
	}
	
	// This is only called outside threading
	void Solver::FlushBuffers()
	{
		for (int x = 0; x < diskBuffer.size(); x++)
		{
			bool needFlush = false;
			for (int t = 0; t < THREADS; t++)
			{
				if (diskBuffer[x].threadData[t].index > 0 || diskBuffer[x].threadData[t].onDisk > 0)
				{
					needFlush = true;
				}
			}
			if (needFlush)
				FlushBuffer(x);
		}
	}
	
	void Solver::FlushBuffer(uint32_t which)
	{
		// Tests on 49/3 show that 20% of the stored states are duplicates
		// Thus, there isn't much overall gain to trying to remove duplicates
		// But, only 8.6% are not duplicates from what is known already!
		// 49/3 numbers 0.0858666 stored: 918,958,141 [Note there are 141,219,540 states, so every state is stored 6.5 times on average
		// 568,158,048 [caching writes]
		data.LoadLargeBufferForWrite(which);
//		std::unordered_map <uint32_t, bool> map;
//		uint64_t count = 0, prov = 0;
		// flush memory
		for (int y = 0; y < THREADS; y++)
		{
			for (int x = 0; x < diskBuffer[which].threadData[y].index; x++)
			{
				auto &i = diskBuffer[which].threadData[y].queue[x];
//				map[(i.baseMemOffset<<24)|i.offset] = true; count++;
				
				uint32_t p1Rank = order[i.baseMemOffset+(which<<8)];
				uint32_t p2Rank = i.offset;
				
				//if (data.SetIf(groups[p1Rank].memoryOffset*r.getMaxP2Rank()+p2Rank, kWin, kDraw))
//				count++;
				if (data.SetIfLargeBuffer(groups[p1Rank].memoryOffset, p2Rank, kWin, kDraw))
				{
//					prov++;
					mProven++;
					groups[p1Rank].lock.lock();
					groups[p1Rank].assignedCount++;
					groups[p1Rank].changed = true;
					groups[p1Rank].lock.unlock();
				}
			}
			diskBuffer[which].threadData[y].index = 0;
		}
		
		// read and clear disk
		fseek(diskBuffer[which].f, 0, SEEK_SET);
		int buffersFromDisk = 0;
		for (int t = 0; t < THREADS; t++)
		{
			for (int y = 0; y < diskBuffer[which].threadData[t].onDisk; y++)
			{
				buffersFromDisk++;
				fread(diskBuffer[which].threadData[t].queue, sizeof(DiskEntry), queueSize, diskBuffer[which].f);
				statesOnDisk -= queueSize;
				for (int x = 0; x < queueSize; x++)
				{
					auto &i = diskBuffer[which].threadData[t].queue[x];
//					map[(i.baseMemOffset<<24)|i.offset] = true; count++;

					uint32_t p1Rank = order[i.baseMemOffset+(which<<8)];
					uint32_t p2Rank = i.offset;
					
					//if (data.SetIf(groups[p1Rank].memoryOffset*r.getMaxP2Rank()+p2Rank, kWin, kDraw))
//					count++;
					if (data.SetIfLargeBuffer(groups[p1Rank].memoryOffset, p2Rank, kWin, kDraw))
					{
//						prov++;
						mProven++;
						groups[p1Rank].lock.lock();
						groups[p1Rank].assignedCount++;
						groups[p1Rank].changed = true;
						groups[p1Rank].lock.unlock();
					}
				}
			}
		}
		
		if (queueSize*buffersFromDisk > 20000) // clear file if it was large
		{
			fclose(diskBuffer[which].f);
			diskBuffer[which].f = fopen(GetTempFileName(which), "w+b"); // open and truncate file to keep it small on disk
		}
		else {
			fseek(diskBuffer[which].f, 0, SEEK_SET);
		}
		for (int t = 0; t < THREADS; t++)
			diskBuffer[which].threadData[t].onDisk = 0;
		
//		if (count != 0)
//			printf("%llu stored; %llu newly proven\n", count, prov);
	}
	
	
	
#pragma mark - other utilities -
	
	void Solver::PrintStats() const
	{
		uint64_t w = 0, l = 0, d = 0, i = 0;
		for (uint64_t x = 0; x < data.Size(); x++)
		{
			switch (data.Get(x))
			{
				case ParallelEM::kWin: w++; break;
				case ParallelEM::kLoss: l++; break;
				case ParallelEM::kIllegal: i++; break;
				case ParallelEM::kDraw: d++; break;
			}
		}
		printf("--Cache Data Summary--\n");
		printf("%llu wins\n%llu losses\n%llu draws\n%llu illegal\n", w, l, d, i);
		std::cout << stat << "\n";

		w = 0;
		l = 0;
		d = 0;
		i = 0;
		
		for (uint64_t x = 0; x < r.getMaxP1Rank(); x++)
		{
			if (groups[x].symmetryRedundant)
				continue;
			
			bool sym = false;
			if (groups[x].symmetricRank != -1)
				sym = true;
			for (uint64_t y = 0; y < r.getMaxP2Rank(); y++)
			{
				switch (data.Get(groups[x].memoryOffset*r.getMaxP2Rank()+y))
				{
					case ParallelEM::kWin: w+=(sym?2:1); break;
					case ParallelEM::kLoss: l+=(sym?2:1); break;
					case ParallelEM::kIllegal: i+=(sym?2:1); break;
					case ParallelEM::kDraw: d+=(sym?2:1); break;
				}
			}
		}
		uint64_t tmp = w;
		w += l;
		l+=tmp;
		d*=2;
		i*=2;
		
		printf("--Cache Data Summary [normalized to full data without symmetry]--\n");
		printf("%llu wins\n%llu losses\n%llu draws\n%llu illegal\n", w, l, d, i);
		std::cout << stat << "\n";		}
	
	const char *Solver::GetFileName(const char *dataPath)
	{
		static std::string s;
		s = dataPath;
		s += "CC-SOLVE-PEM-";
		s += std::to_string(NUM_SPOTS);
		s += "-";
		s += std::to_string(NUM_PIECES);
		s += "-";
		s += r.name();
		return s.c_str();
	}
	
	const char *Solver::GetTempFileName(int which)
	{
		static std::string s;
		s = scratchPath;
		s += "CC-TMP-PEM-"+std::to_string(which);
		s += ".dat";
		return s.c_str();
	}
	
	tResult Solver::Translate(int nextPlayer, tResult res) const
	{
		switch (nextPlayer)
		{
			case 0: return res;
			case 1:
			{
				// 	kWin = 2, kLoss = 1, kDraw = 0, kIllegal = 3
				tResult inv[4] = {kDraw, kWin, kLoss, kIllegal};
				return inv[res];
			}
		}
		assert(false);
		return kIllegal;
	}
	
	tResult Solver::Translate(const CCState &s, tResult res) const
	{
		switch (s.toMove)
		{
			case 0: return res;
			case 1:
			{
				// 	kWin = 2, kLoss = 1, kDraw = 0, kIllegal = 3
				tResult inv[4] = {kDraw, kWin, kLoss, kIllegal};
				return inv[res];
			}
		}
		assert(false);
		return kIllegal;
	}
	
	tResult Solver::Lookup(const CCState &s) const
	{
		static CCheckers cc;
		int64_t p1, p2;
		uint64_t v = r.rank(s, p1, p2);
		if (groups[p1].symmetryRedundant)
		{
			CCState tmp = s;
			cc.SymmetryFlipHoriz(tmp);
			v = r.rank(tmp, p1, p2);
			if (groups[p1].symmetryRedundant)
			{
				printf("Flipped state and still symmetry redundant!\n");
				exit(0);
			}
		}
		return Translate(s, (tResult)data.Get(groups[p1].memoryOffset, p2));
		//		return Translate(s, (tResult)data.Get(v));
	}
	
	void Solver::DebugState(CCState &s)
	{
		CCheckers cc;
		s.PrintASCII();
		CCMove *c = cc.getMoves(s);
		for (CCMove *n = c; n; n = n->next)
		{
			cc.ApplyMove(s, n);
			printf("{%d} ", Lookup(s) );
			s.PrintASCII();
			cc.UndoMove(s, n);
		}
	}
	
	void Solver::DebugState(int64_t r1, int64_t r2)
	{
		CCState s;
		r.unrank(r1, r2, s);
		DebugState(s);
	}

}


