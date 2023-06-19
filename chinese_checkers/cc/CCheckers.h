#include "stdint.h"
#include <iostream>
#include "screenUtil.h"
#include <vector>
#include "/Users/bigyankarki/opt/anaconda3/envs/cc2/include/pybind11/pybind11.h"
#include "/Users/bigyankarki/opt/anaconda3/envs/cc2/include/pybind11/numpy.h"

namespace py = pybind11;

#ifndef __CCHECKERS_H
#define __CCHECKERS_H

//#define BOARD_121_10
// #define BOARD_81_10
//#define BOARD_81_8
//#define BOARD_81_7
//#define BOARD_81_6
//#define BOARD_81_5
// #define BOARD_81_4
//#define BOARD_73_6
// #define BOARD_49_6
//#define BOARD_49_5
//#define BOARD_49_4
//#define BOARD_49_3
//#define BOARD_49_2
//#define BOARD_49_1
//#define BOARD_36_6
//#define BOARD_36_4
#define BOARD_25_6
//#define BOARD_16_6
//#define BOARD_16_4
// #define BOARD_16_3
//#define BOARD_9_3
//#define BOARD_9_1
//#define BOARD_4_1

#ifdef BOARD_121_10 // full board, 10 pieces
#define DIAGONAL_LEN 9
#define NUM_SPOTS 121
#define NUM_PIECES 10
#define ROWS 17
#define COLUMNS 25
#endif

#ifdef BOARD_81_10 // full board, 7 pieces
#define DIAGONAL_LEN 9
#define NUM_SPOTS 81
#define NUM_LEFT_SIDE 45
#define NUM_PIECES 10
#define ROWS 17
#define COLUMNS 25
#endif

#ifdef BOARD_81_8 // full board, 7 pieces
#define DIAGONAL_LEN 9
#define NUM_SPOTS 81
#define NUM_LEFT_SIDE 45
#define NUM_PIECES 8
#define ROWS 17
#define COLUMNS 25
#endif

#ifdef BOARD_81_7 // full board, 7 pieces
#define DIAGONAL_LEN 9
#define NUM_SPOTS 81
#define NUM_LEFT_SIDE 45
#define NUM_PIECES 7
#define ROWS 17
#define COLUMNS 25
#endif

#ifdef BOARD_81_6 // full board, 6 pieces
#define DIAGONAL_LEN 9
#define NUM_SPOTS 81
#define NUM_LEFT_SIDE 45
#define NUM_PIECES 6
#define ROWS 17
#define COLUMNS 25
#endif

#ifdef BOARD_81_5 // full board, 5 pieces
#define DIAGONAL_LEN 9
#define NUM_SPOTS 81
#define NUM_LEFT_SIDE 45
#define NUM_PIECES 5
#define ROWS 17
#define COLUMNS 25
#endif

#ifdef BOARD_81_4 // full board, 4 pieces
#define DIAGONAL_LEN 9
#define NUM_SPOTS 81
#define NUM_LEFT_SIDE 45
#define NUM_PIECES 4
#define ROWS 17
#define COLUMNS 25
#endif

#ifdef BOARD_73_6 // small board, 6 pieces
#define NUM_SPOTS 73
#define NUM_PIECES 6
#define ROWS 13
#define COLUMNS 21
#endif

#ifdef BOARD_49_6 // small board, 6 pieces
#define DIAGONAL_LEN 7
#define NUM_SPOTS 49
#define NUM_LEFT_SIDE 28
#define NUM_PIECES 6
#define ROWS 13
#define COLUMNS 21
#endif

#ifdef BOARD_49_5 // small board, 5 pieces
#define DIAGONAL_LEN 7
#define NUM_SPOTS 49
#define NUM_LEFT_SIDE 28
#define NUM_PIECES 5
#define ROWS 13
#define COLUMNS 21
#endif

#ifdef BOARD_49_4 // small board, 4 pieces
#define DIAGONAL_LEN 7
#define NUM_SPOTS 49
#define NUM_LEFT_SIDE 28
#define NUM_PIECES 4
#define ROWS 13
#define COLUMNS 21
#endif

#ifdef BOARD_49_3 // small board, 3 pieces
#define DIAGONAL_LEN 7
#define NUM_SPOTS 49
#define NUM_LEFT_SIDE 28
#define NUM_PIECES 3
#define ROWS 13
#define COLUMNS 21
#endif

#ifdef BOARD_49_2 // small board, 2 pieces, no extra star corners
#define DIAGONAL_LEN 7
#define NUM_SPOTS 49
#define NUM_LEFT_SIDE 28
#define NUM_PIECES 2
#define ROWS 13
#define COLUMNS 21
#endif

#ifdef BOARD_49_1 // small board, 2 pieces, no extra star corners
#define DIAGONAL_LEN 7
#define NUM_SPOTS 49
#define NUM_LEFT_SIDE 28
#define NUM_PIECES 1
#define ROWS 13
#define COLUMNS 21
#endif

#ifdef BOARD_36_6 // smaller board, 6 pieces
#define DIAGONAL_LEN 6
#define NUM_SPOTS 36
#define NUM_LEFT_SIDE 21
#define NUM_PIECES 6
#define ROWS 11
#define COLUMNS 21
#endif

#ifdef BOARD_36_4 // smaller board, 4 pieces
#define DIAGONAL_LEN 6
#define NUM_SPOTS 36
#define NUM_LEFT_SIDE 21
#define NUM_PIECES 4
#define ROWS 11
#define COLUMNS 21
#endif

#ifdef BOARD_25_6 // smaller board, 6 pieces
#define DIAGONAL_LEN 5
#define NUM_SPOTS 25
#define NUM_LEFT_SIDE 15
#define NUM_PIECES 6
#define ROWS 9
#define COLUMNS 21
#endif

#ifdef BOARD_16_6 // smaller board, 6 pieces
#define DIAGONAL_LEN 4
#define NUM_SPOTS 16
#define NUM_LEFT_SIDE 10
#define NUM_PIECES 6
#define ROWS 7
#define COLUMNS 21
#endif

#ifdef BOARD_16_4 // smaller board, 3 pieces
#define DIAGONAL_LEN 4
#define NUM_SPOTS 16
#define NUM_LEFT_SIDE 10
#define NUM_PIECES 4
#define ROWS 7
#define COLUMNS 21
#endif

#ifdef BOARD_16_3 // smaller board, 3 pieces
#define DIAGONAL_LEN 4
#define NUM_SPOTS 16
#define NUM_LEFT_SIDE 10
#define NUM_PIECES 3
#define ROWS 7
#define COLUMNS 21
#endif


#ifdef BOARD_9_3
#define DIAGONAL_LEN 3
#define NUM_SPOTS 9
#define NUM_LEFT_SIDE 6
#define NUM_PIECES 3
#define ROWS 5
#define COLUMNS 19
#endif


#ifdef BOARD_9_1
// small board, 1 piece, no extra star corners
#define DIAGONAL_LEN 3
#define NUM_SPOTS 9
#define NUM_LEFT_SIDE 6
#define NUM_PIECES 1
#define ROWS 5
#define COLUMNS 19
#endif

#ifdef BOARD_4_1
// small board, 1 piece, no extra star corners
#define DIAGONAL_LEN 2
#define NUM_SPOTS 4
#define NUM_LEFT_SIDE 3
#define NUM_PIECES 1
#define ROWS 3
#define COLUMNS 17
#endif

#define NUM_PLAYERS 2
//#define goalDistance(gs, p1, plyr) gs->distance(gs->getGoal(plyr), p1)

#define CACHE_SIZE 1000

class CCMove;
class CCheckers;

class CCMove {
public:
	CCMove(CCMove *_next = nullptr)
    :from(0), to(0) { next = _next; }
	CCMove(int f, int t, CCMove *n)
    :from(f), to(t)
    {
		next = n;
    }
	~CCMove() {}
	CCMove *clone(CCheckers &cc) const;
	void Print(int all) { fprintf(stderr, "(%2d, %2d) ", from, to); if (all && next) next->Print(all); }
	int add(CCMove *m); // 1 == success, 0 = failure (duplicate)
	int length() { if (next == 0) return 1; return 1+next->length(); }
	uint8_t from, to, which;
	CCMove *next;
    int getFrom() { return from; }
    int getTo() { return to; }
    int getWhich() { return which; }
    CCMove *getNextMove() { return next; }
    void setNextMove(CCMove *n) { next = n; }
};


// Pieces are always sorted, with the largest piece in position 0 and the smallest in the position NUM_PIECES-1
class CCState {
public:
	void Print() const;
	void PrintASCII() const;
	void Reverse();
	void Verify();

	void SetPiecesFromBoard();
	void SetToMoveFromBoard();


	int board[NUM_SPOTS];
	int pieces[NUM_PLAYERS][NUM_PIECES];
	int toMove;	
    py::array_t<int> getBoard() { return py::array(NUM_SPOTS, board); }
    py::array_t<int> getPieces() { return py::array(NUM_PLAYERS, pieces); }
    int getToMove() { return toMove; }
};

CCState listToCCState(const std::vector<int>& board_list, int toMove);


static inline bool operator==(const CCState& a, const CCState& b)
{
	if (a.toMove != b.toMove)
		return false;
	for (int x = 0; x < NUM_SPOTS; x++)
		if (a.board[x] != b.board[x])
			return false;
	return true;
}

static inline bool operator!=(const CCState& a, const CCState& b)
{
	return !(a == b);
}


class CCheckers {
public:
	CCheckers();
	~CCheckers();
	int GetMoveCacheSize();
	void Reset(CCState &state);

	void ResetP1Goal(CCState &state) const;
	void ResetP1(CCState &state, int64_t &p1Rank) const;
	bool SetP2Goal(CCState &state) const;
	bool IncrementP1(CCState &state, int64_t &p1Rank) const;
	void ResetP1RelP2(CCState &state) const;
	bool IncrementP1RelP2(CCState &state) const;

    void applyState(const std::string configuration, CCState &state);
    void delMove(CCMove *m);

	int GetNextFreeLoc(CCState &s, int start) const;
	/*static*/ void ResetGlobals();
	void setState(CCState &s, int p1[NUM_PIECES], int p2[NUM_PIECES]);
	int getNextPlayerNum() const;
	int getPreviousPlayerNum() const;
	void ApplyMove(CCState &s, CCMove *move);
	void UndoMove(CCState &s, CCMove *move);
	void ApplyReverseMove(CCState &s, CCMove *move);
	void UndoReverseMove(CCState &s, CCMove *move);
	CCMove *getMoves(const CCState &s) const;
	CCMove *getMovesForward(const CCState &s) const;
	CCMove *getReverseMoves(CCState &s);
	CCMove *getMovesForPiece(CCState &s, int which);
	
	int getGoal(int who) const;
	int getStart(int who) const;
	
	bool Done(const CCState &s) const;
	int Winner(const CCState &s) const;
	bool Legal(const CCState &s) const;
	int GetNumPiecesInGoal(const CCState &s, int who) const;
	int GetNumPiecesInStart(const CCState &s, int who) const;
	int64_t getMaxRank() const;
	int64_t rank(const CCState &s) const;
	// returns true if it is a valid unranking
	bool unrank(int64_t, CCState &s) const;

	int64_t getMaxSinglePlayerRank() const;
	int64_t getMaxSinglePlayerRank2() const;
	int64_t getMaxSinglePlayerRank2(int64_t firstIndex) const;

	int64_t rankPlayer(const CCState &s, int who) const;
	void rankPlayer(const CCState &s, int who, int64_t &index1, int64_t &index2) const;
	void rankPlayerFirstTwo(CCState &s, int who, int64_t &index1) const;
	void rankPlayerRemaining(CCState &s, int who, int64_t &index2) const;
	int64_t rankPlayerRelative(const CCState &s, int who, int relative) const;
	// Ranks p1 and p2 simultaneously
	void rank(const CCState &s, int64_t &p1, int64_t &p2) const;

	int64_t rankPlayerFlipped(const CCState &s, int who) const; // TODO: write these
	void rankPlayerFlipped(const CCState &s, int who, int64_t &index1, int64_t &index2) const;
	void rankPlayerFirstTwoFlipped(CCState &s, int who, int64_t &index1) const;
	void rankPlayerRemainingFlipped(CCState &s, int who, int64_t &index2) const;

	
	bool twoPieceNeighbors(int64_t i1, int64_t i2);
	// returns true if it is a valid unranking given existing pieces
	bool unrankPlayer(int64_t, CCState &s, int who) const;
	bool unrankPlayer(int64_t, int64_t, CCState &s, int who);
	bool unrankPlayerFirstTwo(int64_t index1, CCState &s, int who);
	bool unrankPlayerRelative(int64_t, CCState &s, int who, int relative) const;
	void unrankPlayerRelativeHelper(int64_t r, CCState &s, int who) const;

	bool MovePlayerToGoal(CCState &s, int who);

	double eval(const CCState &s, int who);
	
	/*static*/ int toxy(int val, int &x, int &y) const;
	/*static*/ int toxy_fast(int val, int &x, int &y) const; // this function doesn't do error checking
	/*static*/ int toxydiamond(int val, int &x, int &y);
	/*static*/ int fromxy(int x, int y) const;
	/*static*/ int fromxytodiamond(int x, int y);
	//static int fromxyto(int x, int y);
	/*static*/ int torc(int val, int &r, int &c) const;
	/*static*/ int fromrc(int r, int c);
	/*static*/ int distance(int, int, int, int) const;
	/*static*/ int distance(int, int) const;
	/*static*/ int newdistance(int, int);
	/*static*/ int lineoffset(int, int, int, int, int, int); // a point and a line (2 pts)
	/*static*/ int rotate60clockwise(int);
	/*static*/ void rotate60clockwise(int &x, int &y);
	int goalOffset(int pos, int plyr);
	int goalDistance(CCState &s, int who) const;
	int startDistance(CCState &s, int who) const;
	
//private:
	
	int64_t getMaxSinglePlayerRank(int numPieces) const;
	int64_t getMaxSinglePlayerRankRelative() const;
	void rankPlayerFirstTwo(CCState &s, int who, int64_t &index1, int numPieces);
	void rankPlayerFirstTwoInFullBoard(CCState &s, int who, int64_t &index1, int numPieces);
	bool unrankPlayer(int64_t rank, CCState &s, int who, int numPieces) const;
	bool validInFullBoard(const CCState &s, int numPieces);
	
	int globalsReset;
	short count[ROWS+1];
	short mapx[NUM_SPOTS];
	short mapy[NUM_SPOTS];
	short reverseMap[COLUMNS][ROWS];
	short places[NUM_PLAYERS][NUM_PIECES];
	short placement[NUM_PLAYERS][NUM_PLAYERS];
//	short goals[NUM_PLAYERS][NUM_PLAYERS];
	short distances[NUM_SPOTS][NUM_SPOTS];
	short leftID[NUM_SPOTS];
	int stepLocations[NUM_SPOTS][6];
	int hopLocations[NUM_SPOTS][6];
	
//public:
	int getCenterlineOffset(int);
	int flipHorizonally(int);
	bool leftHalf(int);
	bool rightHalf(int);
	bool symmetricStart(CCState &s);
	bool specialStart(CCState &s);
	int getLeftID(int location);
//private:
	
	std::vector<int64_t> theSums;
	std::vector<int64_t> binomials;
	std::vector<int64_t> rankOffsets;

	int64_t multinomial(unsigned int n, unsigned int k1, unsigned int k2) const;
	int64_t bi(unsigned int n, unsigned int k) const;
	int64_t binomial(unsigned int n, unsigned int k) const;
	int64_t binomialSum(unsigned int n1, unsigned int n2, unsigned int k) const;
	int64_t biSum(unsigned int n1, unsigned int n2, unsigned int k);
	void initBinomialSums();
	void initBinomial();

	int flip[NUM_SPOTS];
	void initFlip();
	void SymmetryFlipHoriz(CCState &s) const;
	void SymmetryFlipVert(CCState &s) const;
	void SymmetryFlipVertP1(CCState &s) const;
	void SymmetryFlipHorizVert(CCState &s) const;
	void SymmetryFlipHoriz(const CCState &s, CCState &result) const;
	void SymmetryFlipVert(const CCState &s, CCState &result) const;
	void SymmetryFlipHorizVert(const CCState &s, CCState &result) const;
	// Doesn't swap the board; only useful for ranking, results in invalid board state
	void SymmetryFlipHoriz_PO(CCState &s) const;
	void SymmetryFlipVert_PO(CCState &s) const;
	void SymmetryFlipHoriz_PO(const CCState &s, CCState &tmp) const;

	void FlipPlayer(const CCState &src, CCState &dest, int who) const;

	
	CCMove *allocateMoreMoves(int);
	CCMove *getSteps(int who, int where, int, const int *, bool forward) const;
	void getHops(int who, int where, int, int, const int *, CCMove*, bool forward) const;
	
	/*static*/ int getQuad(int x, int y);
	
	/*static*/ int realtoxy(int val, int &x, int &y);
	/*static*/ int realfromxy(int x, int y);
//public:
	CCMove *getNewMove() const;
	void freeMove(CCMove *m) const;
private:
	mutable CCMove *moveCache;
	CCMove realCache[CACHE_SIZE];
	std::vector<bool> forbidden;
	mutable int64_t forbidden1, forbidden2;
};

//int64_t binomialSum(unsigned int n1, unsigned int n2, unsigned int k);
//int64_t binomial(unsigned int n, unsigned int k);
//int64_t multinomial(unsigned int n, unsigned int k1, unsigned int k2);



static inline std::ostream& operator <<(std::ostream & out, const CCMove &move)
{
	out << "(" << (int)move.from << "->" << (int)move.to << ")";
//	if (move.next)
//		out << move.next;
	return out;
}

#endif
