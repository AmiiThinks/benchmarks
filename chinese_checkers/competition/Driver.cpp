//
//  Driver.cpp
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/17/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#include <unordered_map>
#include <sstream>
#include <string>
#include "CCUtils.h"
#include "Driver.h"
#include "Minimax.h"
#include "MinimaxDB.h"
#include "DistEval.h"
#include "LBDistEval.h"
#include "DistDBEval.h"
#include "TDRegression.h"
#include "UCB.h"
#include "UCT.h"
#include "CCEndgameData.h"
#include "DBPlusEval.h"
#include "UCTDB.h"

std::string name = "Professor";
std::string opponent;
const double timeLimit = 9.8;
int depthLimit = 1000;

//CCMove *nextMove() {
//	// Somehow select your next move
//	std::vector<Move> moves;
//	state.getMoves(moves);
//	return moves[0];
//}

int LocalToServer(int val)
{
	const int table[81] =
	{0, 9, 1, 18, 10, 2, 27, 19, 11, 3, 36, 28, 20, 12, 4, 45, 37, 29, 21, 13, 5, 54, 46, 38, 30, 22, 14, 6, 63, 55, 47, 39, 31, 23, 15, 7, 72, 64, 56, 48, 40, 32, 24, 16, 8, 73, 65, 57, 49, 41, 33, 25, 17, 74, 66, 58, 50, 42, 34, 26, 75, 67, 59, 51, 43, 35, 76, 68, 60, 52, 44, 77, 69, 61, 53, 78, 70, 62, 79, 71, 80};
	return table[val];
}

int ServerToLocal(int val)
{
	const int table[81] =
	{0, 2, 5, 9, 14, 20, 27, 35, 44, 1, 4, 8, 13, 19, 26, 34, 43, 52, 3, 7, 12, 18, 25, 33, 42, 51, 59, 6, 11, 17, 24, 32, 41, 50, 58, 65, 10, 16, 23, 31, 40, 49, 57, 64, 70, 15, 22, 30, 39, 48, 56, 63, 69, 74, 21, 29, 38, 47, 55, 62, 68, 73, 77, 28, 37, 46, 54, 61, 67, 72, 76, 79, 36, 45, 53, 60, 66, 71, 75, 78, 80};
	return table[val];
}

// Reads a line, up to a newline from the server
std::string readMsg()
{
	std::string msg;
	std::getline(std::cin, msg); // This is a blocking read
	
	// Trim white space from beginning of string
	const char *WhiteSpace = " \t\n\r\f\v";
	msg.erase(0, msg.find_first_not_of(WhiteSpace));
	// Trim white space from end of string
	msg.erase(msg.find_last_not_of(WhiteSpace) + 1);
	
	return msg;
}

// Sends a msg to stdout and verifies that the next message to come in is it
// echoed back. This is how the server validates moves
void printAndRecvEcho(CCMove *m)
{
	std::string message = "MOVE FROM "+std::to_string(LocalToServer(m->from))+" TO "+std::to_string(LocalToServer(m->to));
	// Note the endl flushes the stream, which is necessary
	std::cout << message << "\n";
	const std::string echo_recv = readMsg();
	if (message != echo_recv)
		std::cerr << "Expected echo of '" << message << "'. Received '" << echo_recv
		<< "'" << std::endl;
}

// Tokenizes a message based upon whitespace
std::vector<std::string> tokenizeMsg(const std::string &msg)
{
	// Tokenize using whitespace as a delimiter
	std::stringstream ss(msg);
	std::istream_iterator<std::string> begin(ss);
	std::istream_iterator<std::string> end;
	std::vector<std::string> tokens(begin, end);
	
	return tokens;
}

int waitForStart() {
	int us;
	for (;;) {
		std::string response = readMsg();
		std::vector<std::string> tokens = tokenizeMsg(response);
		
		if (tokens.size() == 4 && tokens[0] == "BEGIN" &&
			tokens[1] == "CHINESECHECKERS") {
			// Found BEGIN GAME message, determine if we play first
			if (tokens[2] == name) {
				// We go first!
				opponent = tokens[3];
				us = 0;
				break;
			} else if (tokens[3] == name) {
				// They go first
				opponent = tokens[2];
				us = 1;
				break;
			} else {
				std::cerr << "Did not find '" << name
				<< "', my name, in the BEGIN command.\n"
				<< "# Found '" << tokens[2] << "' and '" << tokens[3] << "'"
				<< " as player names. Received message '" << response << "'"
				<< std::endl;
				std::cout << "#quit" << std::endl;
				std::exit(EXIT_FAILURE);
			}
		} else if (response == "DUMPSTATE") {
			//std::cout << state.dumpState() << std::endl;
		} else if (tokens[0] == "LOADSTATE") {
//			std::string new_state = response.substr(10);
//			if (!state.loadState(new_state))
//				std::cerr << "Failed to load '" << new_state << "'\n";
		} else if (response == "LISTMOVES") {
//			std::vector<Move> moves;
//			state.getMoves(moves);
//			for (const auto i : moves)
//				std::cout << i.from << ", " << i.to << "; ";
//			std::cout << std::endl;
		} else if (tokens[0] == "MOVE") {
//			// Just apply the move
//			const Move m = state.translateToLocal(tokens);
//			if (!state.applyMove(m))
//				std::cout << "Unable to apply move " << m << std::endl;
		} else if (tokens[0] == "UNDO") {
//			tokens[0] = "MOVE";
//			const Move m = state.translateToLocal(tokens);
//			if (!state.undoMove(m))
//				std::cout << "Unable to undo move " << m << std::endl;
		} else if (response == "NEXTMOVE") {
//			const Move m = nextMove();
//			std::cout << m.from << ", " << m.to << std::endl;
		} else if (response == "EVAL") {
//			// somehow evaluate the state like: eval(state, state.getCurrentPlayer())
//			std::cout << "0.00" << std::endl;
		} else {
			std::cerr << "Unexpected message " << response << "\n";
		}
	}
	
	// Game is about to begin, restore to start state in case DUMPSTATE/LOADSTATE/LISTMOVES
	// were used
	//state.reset();
	
	// Player 1 goes first
	return us;
}

TDEval Train()
{
	TDRegression tdr(.05, 0.8);
	std::vector<Player *> p;
	std::vector<CCMove *> moves;
	p.push_back(&tdr);
	p.push_back(&tdr);
	
	CCheckers cc;
	CCState s;
	for (int x = 0; x < 50000; x++)
	{
		cc.Reset(s);
		std::swap(p[0], p[1]);
		while (!cc.Done(s))
		{
			double value;
			//s.Print();
			CCMove *next = p[s.toMove]->GetNextAction(&cc, s, value, 0.05);
			cc.ApplyMove(s, next);
			moves.push_back(next);
		}
		tdr.Train(&cc, moves);
		
		while (!moves.empty())
		{
			cc.freeMove(moves.back());
			moves.pop_back();
		}
		if (0 == x%10000)
		{
			fprintf(stderr, "Training round %d done\n", x);
		}
	}
	
	//tdr.Print();
	return tdr.GetTDEval();
}


int main(int argc, char **argv)
{
	
	CCheckers cc;
	CCState s;
	cc.Reset(s);
	int ourPlayer;
	TDEval e = Train();
	DBPlusEval<TDEval> db(&e, dataPrefix, 1962, 36);
	Minimax<DBPlusEval<TDEval>> player(&db, false);

	
	
	std::cout << "#name " << name << std::endl;
	
	// Wait for start of game
	ourPlayer = waitForStart();
	
	// Main game loop
	for (;;) {
		if (s.toMove == ourPlayer) {
			// My turn
			
			// Check if game is over
			if (cc.Done(s))
			{
				if (cc.Winner(s) == ourPlayer)
				{
					std::cerr << "I, " << name << ", have won" << std::endl;
				}
				else {
					std::cerr << "I, " << name << ", have lost" << std::endl;
				}
			}
			
			// Determine next move
			double score;
			CCMove *m = player.GetNextAction(&cc, s, score, timeLimit, depthLimit);

			// Apply it locally
			cc.ApplyMove(s, m);

			// Tell the world
			printAndRecvEcho(m);
			
			cc.freeMove(m);
			
		} else {
			// Wait for move from other player
			// Get server's next instruction
			std::string server_msg = readMsg();
			const std::vector<std::string> tokens = tokenizeMsg(server_msg);
			
			if (tokens.size() == 5 && tokens[0] == "MOVE") {
				int from = ServerToLocal(atoi(tokens[2].c_str()));
				int to = ServerToLocal(atoi(tokens[4].c_str()));

				// Translate to local coordinates and update our local state
				bool applied = false;
				CCMove *m = cc.getMoves(s);
				for (CCMove *t = m; t; t = t->next)
				{
					if (t->from == from && t->to == to)
					{
						cc.ApplyMove(s, t);
						applied = true;
						break;
					}
				}
				cc.freeMove(m);
				if (!applied)
				{
					std::cerr << "Couldn't find move from " << from << " to " << to << "\n";
				}
			}
			else if (tokens.size() == 4 && tokens[0] == "FINAL" &&
					   tokens[2] == "BEATS")
			{
				// Game over
				if (tokens[1] == name && tokens[3] == opponent)
					std::cerr << "I, " << name << ", have won!" << std::endl;
				else if (tokens[3] == name && tokens[1] == opponent)
					std::cerr << "I, " << name << ", have lost." << std::endl;
				else
					std::cerr << "Did not find expected players in FINAL command.\n"
					<< "Found '" << tokens[1] << "' and '" << tokens[3] << "'. "
					<< "Expected '" << name << "' and '" << opponent << "'.\n"
					<< "Received message '" << server_msg << "'" << std::endl;
				break;
			}
			else {
				// Unknown command
				std::cerr << "Unknown command of '" << server_msg << "' from the server"
				<< std::endl;
			}
		}
	}
	
	return 0;
}
