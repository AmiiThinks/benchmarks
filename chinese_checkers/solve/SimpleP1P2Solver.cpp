//
//  EMSolver.cpp
//  CC Solver
//
//  Created by Nathan Sturtevant on 11/5/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#include "SimpleP1P2Solver.h"
#include <cassert>
#include <cstring>
#include <thread>
#include <array>
#include <unordered_map>
#include "Timer.h"

namespace SimpleP1P2Solver {
	static const char *resultText[] = {"Draw", "Loss", "Win", "Illegal"};

	const int SimpleP1P2Solver::Solver::kNumGroups;
	const int SimpleP1P2Solver::Solver::kEntriesPerGroup;
	//const int SimpleP1P2Solver::Solver::kNumFiles;

	const bool doParentPropagation = true;

#pragma mark - Initialization -

	Solver::Solver(const char *dataPath, const char *scratchPath, bool forceBuild)
	{
		if (NUM_PIECES == 2 || NUM_PIECES == 5)
		{
			printf("%d pieces not symmetric; aborting\n", NUM_PIECES);
			exit(0);
		}
		printf("### Chinese Checkers Solver! Board: %d Pieces: %d\n", NUM_SPOTS, NUM_PIECES);
		printf("--> Starting SimpleP1P2Solver Solver [%s] --\n", __FILE__);
		printf("--> Ranking: %s --\n", r.name());

		
		InitMetaData();
		printf("--> Done initializing meta-data.\n");
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
		assert(p1p2groups.size() == r.getMaxP1Rank());
		//groups.resize(r.getMaxP1Rank());
		
		// Note that in single-player CC we have to (maybe?) generate
		// the children of both the regular and flipped states
		
		int64_t totalStates = 0;
		symmetricStates = 0;
		for (int32_t x = 0; x < p1p2groups.size(); x++)
			p1p2groups[x].symmetricRank = -1;
		for (int32_t x = 0; x < p1p2groups.size(); x++)
		{
			p1p2groups[x].changed = 0;
			p1p2groups[x].assignedCount = 0;
			r.unrank(x, 0, s);
			CCState sym;
			cc.FlipPlayer(s, sym, 0);
			
			// Since we are only going up to 7x7 (6 piece) CC, we can narrow to 32 bits here
			int32_t otherRank = static_cast<int32_t>(r.rankP1(sym));
			if (otherRank < x)
			{
				p1p2groups[x].symmetryRedundant = true;
				p1p2groups[x].symmetricRank = otherRank;
				p1p2groups[otherRank].symmetricRank = x;
			}
			else {
				p1p2groups[x].memoryOffset = memoryOffset;
				memoryOffset++;//= r.getMaxP2Rank();
				p1p2groups[x].symmetryRedundant = false;
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
		p1p2data.Resize(symmetricStates);
		
		memoryOffset = 0;
		printf("--> Constructing ordering\n");
		for (int32_t x = 0; x < order.size(); x++)
		{
			if (!p1p2groups[order[x]].symmetryRedundant)
			{
				//printf(" [%d] Group %d is at memory %llu\n", x, order[x], memoryOffset);
				p1p2groups[order[x]].memoryOffset = memoryOffset;
				memoryOffset += 1;//r.getMaxP2Rank();
			}
		}
		printf("--> %lu reduced groups [from %lu - %1.3fx]\n", order.size(), p1p2groups.size(), p1p2groups.size()/(float)order.size());
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
			if (!p1p2groups[i.first].symmetryRedundant)
				order.push_back(i.first);
		}
	}
	
	void Solver::GetSearchOrder()
	{
//		printf("--> ORDER: Cache optimized\n");
//		GetCacheOrder();
//		return;
		
		//for (int x = groups.size()-1; x >= 0; x--)
		for (int x = 0; x < p1p2groups.size(); x++)
			if (!p1p2groups[x].symmetryRedundant)
				order.push_back(x);
		std::reverse(order.begin(), order.end());
		printf("--> ORDER: default (reversed)\n");
		return;
		
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
					if (!p1p2groups[x].symmetryRedundant)
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
	
	uint64_t Solver::ThreadInitial()
	{
		// TODO: Current status: verify whether we need to flush buffers below. Then test.
		CCheckers cc;
		CCState s, tmp;
		int64_t max2 = r.getMaxP2Rank();
		int64_t localProven = 0;

		for (int64_t x = 0; x < order.size(); x++)
		{
			int64_t p1Rank = order[x];
			
			if (p1p2groups[p1Rank].symmetryRedundant)
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
				//if (data.SetIf(p1p2groups[p1Rank].memoryOffset*r.getMaxP2Rank(), kLoss, kDraw))
				if (p1p2data.SetIf(p1p2groups[p1Rank].memoryOffset*kEntriesPerGroup+0, kLoss, kDraw))
				{
					p1p2groups[p1Rank].assignedCount++;
					p1p2groups[p1Rank].changed = true;
					// TODO: When this is parallelized, needs to be updated
					localProven++;
					PropagateWinToParent(cc, s); // This is a win at the parent!
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
					auto v = p1p2data.Get(p1p2groups[p1Rank].memoryOffset*kEntriesPerGroup+p2Rank);
					if (v == kDraw)
					{
						// TODO: When this is parallelized, needs to be updated
						localProven++;
						p1p2groups[p1Rank].assignedCount++;
						p1p2groups[p1Rank].changed = true;
					}
					p1p2data.Set(p1p2groups[p1Rank].memoryOffset*kEntriesPerGroup+p2Rank, kIllegal);
					//data.Set(p1p2groups[p1Rank].memoryOffset*r.getMaxP2Rank()+p2Rank, kIllegal);
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
						if (p1p2data.SetIf(p1p2groups[p1Rank].memoryOffset*kEntriesPerGroup+p2Rank, kLoss, kDraw))
						{
							// TODO: When this is parallelized, needs to be updated
							localProven++;
							p1p2groups[p1Rank].assignedCount++;
							p1p2groups[p1Rank].changed = true;
							tmp = s;
							PropagateWinToParent(cc, tmp); // This is a win at the parent!
						}
						else if (p1p2data.SetIf(p1p2groups[p1Rank].memoryOffset*kEntriesPerGroup+p2Rank, kLoss, kWin))
						{
							//proven++;
							//p1p2groups[p1Rank].assignedCount++;
							p1p2groups[p1Rank].changed = true;
							tmp = s;
							PropagateWinToParent(cc, tmp); // This is a win at the parent!
						}
						break;
				}
			}
		}
		return localProven;
	}
//	SharedQueue<int> workUnits;

	void Solver::Initial()
	{
		mProven += ThreadInitial();
	}
	
#pragma mark - main solver -
	
	/*
	 * Note that s will be modified in this function
	 */
	void Solver::PropagateWinToParent(CCheckers &cc, CCState &s)
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
			if (!p1p2groups[p1Rank].symmetryRedundant)
			{
				p2Rank = r.rankP2(s);
				WriteToBuffer(p1Rank, p2Rank, kWin);
			}
			
			// Only flip pieces
			cc.SymmetryFlipHoriz_PO(s, tmp);
			
			p1Rank = r.rankP1(tmp);
			
			stat.rank++;
			if (!p1p2groups[p1Rank].symmetryRedundant)
			{
				p2Rank = r.rankP2(tmp);
				WriteToBuffer(p1Rank, p2Rank, kWin);
			}
			cc.UndoReverseMove(s, t);
		}
		cc.freeMove(m);
	}
	
	
	/*
	 * Returns the value for the parent, so it has to be flipped.
	 */
	tResult Solver::GetValue(CCheckers &cc, const CCState &s, int64_t finalp1Rank, bool doubleFlip)
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
			//tResult tmpVal = (tResult)data.GetReadOnlyBuffer(p1p2groups[p1Rank].memoryOffset, p2Rank, readBuffer);
			tResult tmpVal = (tResult)p1p2data.Get(p1p2groups[p1Rank].memoryOffset*kEntriesPerGroup+p2Rank);
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
	
	uint64_t Solver::SinglePassInnerLoop(CCheckers &cc, CCState s, int64_t r1, int64_t finalp1Rank, bool doubleFlip)
	{
		uint64_t proven = 0;
		assert(s.toMove == kMaxPlayer);
		if (p1p2groups[r1].symmetryRedundant)
		{
			// We can look up the children, but we can't store the parent value. Skip.
			return proven;
		}
		int64_t r2 = r.rankP2(s);

		// Check whether we need to prove this state
		if (p1p2data.Get(p1p2groups[r1].memoryOffset*kEntriesPerGroup+r2) != kDraw)
			return proven;
		
		tResult result = GetValue(cc, s, finalp1Rank, doubleFlip);
		
		switch (result)
		{
			case kWin:
				// Normally we don't see wins at max nodes, since they are immediately propagated. But, with delayed
				// processing of parents we might miss some.
				//if (data.SetIf(p1p2groups[r1].memoryOffset*r.getMaxP2Rank()+r2, kWin, kDraw))
				if (p1p2data.SetIf(p1p2groups[r1].memoryOffset*kEntriesPerGroup+r2, kWin, kDraw))
				{
					p1p2groups[r1].assignedCount++;
					p1p2groups[r1].changed = true;
					proven++;
				}
				break;
			case kLoss:
				// Loss symmetrically becomes win at parent
				//if (data.SetIf(groups[r1].memoryOffset*r.getMaxP2Rank()+r2, kLoss, kDraw))
				if (p1p2data.SetIf(p1p2groups[r1].memoryOffset*kEntriesPerGroup+r2, kLoss, kDraw))
				{
					PropagateWinToParent(cc, s);
					p1p2groups[r1].assignedCount++;
					p1p2groups[r1].changed = true;
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
	
	void Solver::ThreadMain(int64_t p1Rank)
	{
//		uint8_t readBuffer[data.GetReadBufferSize()];
		CCheckers cc;
//		while (true)
//		{
//			// 1. get work - provides the p1Rank being done this round
//			int64_t p1Rank;
//			uint64_t proven;
//			uint64_t totalProven = 0;
//			work[whichThread].WaitRemove(p1Rank);
//			if (p1Rank == -1)
//				return;
//			data.CopyReadOnlyBuffer(readBuffer);
//			// 2. get individual work units
//			int whichFile;
//			while (true) {
//				workUnits.WaitRemove(whichFile);
//				if (whichFile == -1)
//					break;
//				DoThreadLoop(cc, p1Rank, proven);
//				totalProven += proven;
//			}
//			results[whichThread].WaitAdd(totalProven);
//		}
		uint64_t proven;
		DoThreadLoop(cc, p1Rank, proven);
		mProven += proven;
	}
//#include <sys/kdebug_signpost.h>

	void Solver::DoThreadLoop(CCheckers &cc, int64_t p1Rank, uint64_t &proven)
	{
		CCState s;
		proven = 0;
		int64_t firstRank = 0;//whichFile*kGroupsPerFile;
		int64_t lastRank = kNumGroups; //std::min((whichFile+1)*kGroupsPerFile, kNumGroups);
		int64_t r1;//, r2;
//		kdebug_signpost_start(3, 0, 0, 0, 2);
		
		// There are max2 possible ranks, but these are interleaved depending on the p2 pieces
		// So, we need to be able to jump to the correct rank
		for (int64_t x = firstRank; x < lastRank; x++)
		{
			if (p1p2groups[order[x]].assignedCount == kEntriesPerGroup)
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
				proven += SinglePassInnerLoop(cc, s, r1, p1Rank, false);
			}
			
			if (p1p2groups[p1Rank].symmetricRank != -1) // There is a symmetric group that leads here
			{
				// Start with symmetric rank and any p2 state
				//r.unrank(groups[p1Rank].symmetricRank, max2-1, s);
				// Start with symmetric rank and no p2 state
				r.unrankP1(p1p2groups[p1Rank].symmetricRank, s);
				// second player will now have symmetric->p1Rank after next move
				cc.SymmetryFlipVertP1(s);
				s.toMove = kMaxPlayer;

				if (r.TryAddR1(s, order[x])) // success
				{
					r1 = order[x];
//					r1 = r.rankP1(s);
//					assert(r1 == order[x]);
					proven += SinglePassInnerLoop(cc, s, r1, p1Rank, true);
				}
			}

		}
//		kdebug_signpost_end(3, 0, 0, 0, 2);

	}

	void Solver::DoLoops(CCheckers &cc, int64_t max2, int64_t p1Rank)
	{
//		// Flush any buffers that are in memory
//		if (doParentPropagation)
//		{
////			kdebug_signpost_start(1, 0, 0, 0, 0);
//			FlushBuffersInMemory();
//
//			// 1. Load (if needed) the chunk [p1Rank] we are currently reading from
//			// 2. Pull any cached states from disk into memory
//
//			// temp disk usage is too big
//			//
//			if ((statesOnDisk*4) > (symmetricStates/4)*1.5)
//			{
//				auto i = statesOnDisk;
//				int cnt = 0;
//				while ((statesOnDisk*4) > (symmetricStates/4)*0.5)
//				{
//					cnt++;
//					FlushBiggestBuffer(); // TODO: Print which buffers are being flushed. Maybe compute duplicates?
//				}
//				printf("[f%llu-%d]\n", i-statesOnDisk, cnt);
//			}
////			else {//if ((statesOnDisk*4) > (symmetricStates/4)*0.25)
////				ConditionalFlushBiggestBuffer();
////			}
////			kdebug_signpost_end(1, 0, 0, 0, 0);
//		}

		//FlushBuffer(p1p2groups[p1Rank].memoryOffset>>8);

		// [??] Read is only valid until we flush buffers
//		data.LoadLargeBufferForWrite(groups[p1Rank].memoryOffset>>8);
//		data.LoadReadOnlyBuffer(groups[p1Rank].memoryOffset);
		
//		kdebug_signpost_start(2, 0, 0, 0, 1);
		// Tell all the threads we are doing this p1Rank
//		for (int thread = 0; thread < THREADS; thread++)
//			work[thread].Add(p1Rank);
//		// send each of the work units across
//		for (int x = 0; x < kNumFiles; x++)
//			workUnits.WaitAdd(x);
//		for (int thread = 0; thread < THREADS; thread++)
//			workUnits.WaitAdd(-1);
//		for (int thread = 0; thread < THREADS; thread++)
//		{
//			uint64_t items;
//			results[thread].WaitRemove(items);
//			mProven += items;
//		}
//		kdebug_signpost_end(2, 0, 0, 0, 1);
		ThreadMain(p1Rank);
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
//				if (doParentPropagation)
//					PrintBufferStats();
			}
			if (!p1p2groups[p1Rank].changed)
			{
				// If nothing in this group changed, then the parent proof can't be
				// changed by re-analyzing these states
				continue;
			}
			if (p1p2groups[p1Rank].symmetryRedundant)
			{
				// No states in this group to analyze
				continue;
			}
			p1p2groups[p1Rank].changed = false;
			
			// Loop over all parents with children in the p1Rank
			DoLoops(cc, max2, p1Rank);
		}
		printf("\n");
	}
	
	bool Solver::NeedsBuild()
	{
		return true;//data.NeedsBuild();
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
		printf("Round 0; %llu new; %llu of %llu proven; %1.2fs elapsed\n", mProven, mProven, symmetricStates, t.GetElapsedTime());
		
//		printf("** Starting up %d threads\n", THREADS);
//		for (int thread = 0; thread < THREADS; thread++)
//		{
//			threads[thread] = std::thread(&Solver::ThreadMain, this, thread);
//		}

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
			printf("Round %d; %llu new; %llu of %llu proven; %1.2fs elapsed\n", iteration, mProven-oldProven, mProven, symmetricStates, t.GetElapsedTime());
			
			if (!startProven && p1p2data.Get(p1p2groups[startR1].memoryOffset*kEntriesPerGroup+startR2) != kDraw)
			{
				printf("Start state proven to be %s\n", resultText[p1p2data.Get(p1p2groups[startR1].memoryOffset*kEntriesPerGroup+startR2)]);
				startProven = true;
			}
			// Make sure we don't have unproven states waiting to be flushed
//			if (mProven == oldProven)
//				FlushBuffers();
		} while (mProven != oldProven);
		totalTime.EndTimer();
		printf("%1.2fs total time elapsed\n", totalTime.GetElapsedTime());
//		printf("%lld states written during proof\n", statesWrittenToDisk);
		//if (!startProven)
		printf("Start state is a %s\n", resultText[p1p2data.Get(p1p2groups[startR1].memoryOffset*kEntriesPerGroup+ startR2)]);

		// TODO: Restore this
		//data.Write(GetFileName());
		PrintStats();
	}

	void Solver::WriteToBuffer(uint32_t group, uint32_t offset, tResult value)
	{
		if (p1p2groups[group].assignedCount == kEntriesPerGroup)
		{
			return;
		}
		if (p1p2data.SetIf(p1p2groups[group].memoryOffset*kEntriesPerGroup+offset, value, kDraw))
		{
			p1p2groups[group].assignedCount++;
			p1p2groups[group].changed = true;
			mProven++; // TODO - parallelize later
		}
		//assert(!"Need to write code here");
	}
	
#pragma mark - other utilities -
	
	void Solver::PrintStats() const
	{
		uint64_t w = 0, l = 0, d = 0, i = 0;
		for (uint64_t x = 0; x < p1p2data.Size(); x++)
		{
			switch (p1p2data.Get(x))
			{
				case SimpleP1P2Solver::kWin: w++; break;
				case SimpleP1P2Solver::kLoss: l++; break;
				case SimpleP1P2Solver::kIllegal: i++; break;
				case SimpleP1P2Solver::kDraw: d++; break;
			}
		}
		printf("--Cache Data Summary--\n");
		printf("%llu wins\n%llu losses\n%llu draws\n%llu illegal\n", w, l, d, i);
		std::cout << stat << "\n";
	}
	
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
		assert(false);
		static std::string s;
		s = "";//scratchPath;
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
		if (p1p2groups[p1].symmetryRedundant)
		{
			CCState tmp = s;
			cc.SymmetryFlipHoriz(tmp);
			v = r.rank(tmp, p1, p2);
			if (p1p2groups[p1].symmetryRedundant)
			{
				printf("Flipped state and still symmetry redundant!\n");
				exit(0);
			}
		}
		return Translate(s, (tResult)p1p2data.Get(p1p2groups[p1].memoryOffset*kEntriesPerGroup+p2));
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


